"""
src/phonetic_trie.py

Banglish Phonetic Encoder, Trie Matcher, and Phonetic Inverted Index for Thikana-Engine.
Provides:
  1. banglish_phonetic_key(word: str) -> tuple[str, str]:
     Canonical phonetic encoder handling both Bangla script and Romanized Banglish.
  2. levenshtein_distance(s1: str, s2: str) -> int:
     Fast pure-Python edit distance implementation.
  3. PhoneticTrie & PhoneticTrieNode:
     In-memory prefix Trie supporting exact, prefix, and fuzzy Levenshtein search.
  4. PhoneticGazetteerMatcher:
     Indexes all 5,285 administrative nodes from BangladeshGazetteer.
     Provides search() and MeasureCollisionRate().
"""

from __future__ import annotations
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from .models import AdminLevel, AdminNode
    from .gazetteer import BangladeshGazetteer
except (ImportError, ValueError):
    from models import AdminLevel, AdminNode
    from gazetteer import BangladeshGazetteer


# -----------------------------------------------------------------------------
# Bangla-to-Latin Transliteration Map
# -----------------------------------------------------------------------------
BANGLA_TO_LATIN: Dict[str, str] = {
    # Independent vowels
    "অ": "a", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
    # Dependent vowel signs (kars)
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
    # Consonants
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "c", "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "f", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "rh", "য়": "y", "ৎ": "t", "ং": "ng", "ঃ": "h", "ঁ": "",
    "্": "",  # Hasanta / virama
    "়": "",  # Nukta sign
}


def transliterate_bangla(text: str) -> str:
    """Transliterates Bangla script characters to Latin phonetic representation."""
    norm = unicodedata.normalize("NFC", text)
    # Canonicalize decomposed nuktas first
    norm = norm.replace("\u09af\u09bc", "য়").replace("\u09a1\u09bc", "ড়").replace("\u09a2\u09bc", "ঢ়")
    res: List[str] = []
    for ch in norm:
        res.append(BANGLA_TO_LATIN.get(ch, ch))
    return "".join(res)



# -----------------------------------------------------------------------------
# Phonetic Key Generation
# -----------------------------------------------------------------------------
def banglish_phonetic_key(word: str) -> Tuple[str, str]:
    """
    Computes canonical Banglish phonetic keys:
      1. Transliterates Bangla characters to canonical Latin representation if needed.
      2. Normalizes aspirated consonants and common Banglish swaps:
         bh->b, ph->f, kh->k, gh->g, chh/ch->c, sh/s->s, z/j->j, v->b, w->o.
      3. Collapses vowel groups into 3 canonical vowel classes:
         ee/ii/i/y -> I
         oo/u -> U
         o/a/e -> A
      4. Deduplicates adjacent identical characters.

    Returns:
      (primary_key_with_vowels, secondary_consonant_skeleton)
      e.g.
        "Mirpur"  -> ("MIRPUR", "MRPR")
        "Meerpur" -> ("MIRPUR", "MRPR")
        "Mirpoor" -> ("MIRPUR", "MRPR")
        "মিরপুর"   -> ("MIRPUR", "MRPR")
        "Badda"   -> ("BADA", "BD")
        "Boda"    -> ("BADA", "BD")
    """
    if not word or not word.strip():
        return ("", "")

    # Clean punctuation and whitespace
    s = word.strip().lower()

    # Step 1: Transliterate Bangla script if present
    if any("\u0980" <= c <= "\u09ff" for c in s):
        s = transliterate_bangla(s)

    # Step 2: Normalize aspirated consonants and common Banglish swaps
    # Order matters: multi-character sequences before single characters
    s = s.replace("chh", "c").replace("ch", "c")
    s = s.replace("kh", "k").replace("gh", "g")
    s = s.replace("th", "t").replace("dh", "d")
    s = s.replace("ph", "f").replace("bh", "b")
    s = s.replace("sh", "s")
    s = s.replace("jh", "j").replace("z", "j")
    s = s.replace("rh", "r")
    s = s.replace("v", "b")
    s = s.replace("w", "o")

    # Step 3: Collapse vowel groups into 3 canonical vowel classes
    # ee/ii/i/y -> I
    # oo/u -> U
    # o/a/e -> A
    s = re.sub(r"ee+|ii+", "I", s)
    s = re.sub(r"oo+", "U", s)
    s = re.sub(r"[iy]", "I", s)
    s = re.sub(r"u", "U", s)
    s = re.sub(r"[oae]", "A", s)

    # Step 4: Deduplicate adjacent identical characters
    dedup: List[str] = []
    for c in s:
        c_upper = c.upper()
        if not dedup or dedup[-1] != c_upper:
            dedup.append(c_upper)
    primary = "".join(dedup)
    primary = re.sub(r"[^A-Z0-9]", "", primary)

    # Step 5: Secondary consonant skeleton (strip canonical vowels I, U, A)
    vowels = {"I", "U", "A"}
    skeleton = "".join(c for c in primary if c not in vowels)

    return (primary, skeleton)



# -----------------------------------------------------------------------------
# Levenshtein Edit Distance
# -----------------------------------------------------------------------------
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculates standard Levenshtein edit distance between two strings in O(min(m, n)) space.
    """
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# -----------------------------------------------------------------------------
# Prefix Trie Implementation
# -----------------------------------------------------------------------------
class PhoneticTrieNode:
    """Node in the Phonetic Trie."""

    def __init__(self) -> None:
        self.children: Dict[str, PhoneticTrieNode] = {}
        self.is_terminal: bool = False
        self.nodes: List[AdminNode] = []


class PhoneticTrie:
    """Character-based prefix Trie storing administrative nodes."""

    def __init__(self) -> None:
        self.root = PhoneticTrieNode()
        self._size = 0

    def insert(self, key: str, node: AdminNode) -> None:
        """Inserts a key and attaches the matching AdminNode to the terminal node."""
        if not key:
            return
        curr = self.root
        for char in key:
            if char not in curr.children:
                curr.children[char] = PhoneticTrieNode()
            curr = curr.children[char]
        curr.is_terminal = True
        if node not in curr.nodes:
            curr.nodes.append(node)
            self._size += 1

    def search_exact(self, key: str) -> List[AdminNode]:
        """Looks up nodes matching the exact key."""
        if not key:
            return []
        curr = self.root
        for char in key:
            if char not in curr.children:
                return []
            curr = curr.children[char]
        return list(curr.nodes) if curr.is_terminal else []

    def search_prefix(self, prefix: str) -> List[AdminNode]:
        """Looks up all nodes matching a key prefix."""
        if not prefix:
            return []
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return []
            curr = curr.children[char]

        results: List[AdminNode] = []
        seen: Set[str] = set()

        def _dfs(n: PhoneticTrieNode) -> None:
            if n.is_terminal:
                for item in n.nodes:
                    if item.id not in seen:
                        seen.add(item.id)
                        results.append(item)
            for child in n.children.values():
                _dfs(child)

        _dfs(curr)
        return results

    def fuzzy_search(
        self, word: str, max_distance: int = 2
    ) -> List[Tuple[AdminNode, int]]:
        """
        Searches the Trie for keys within max_distance Levenshtein edits.
        Returns list of (AdminNode, edit_distance) tuples sorted by distance.
        """
        if not word:
            return []

        word_len = len(word)
        current_row = list(range(word_len + 1))
        results: Dict[str, Tuple[AdminNode, int]] = {}

        def _search_recursive(
            node: PhoneticTrieNode, char: str, prev_row: List[int], current_str: str
        ) -> None:
            columns = word_len + 1
            new_row = [prev_row[0] + 1]

            for c in range(1, columns):
                insert_cost = new_row[c - 1] + 1
                delete_cost = prev_row[c] + 1
                replace_cost = prev_row[c - 1] + (word[c - 1] != char)
                new_row.append(min(insert_cost, delete_cost, replace_cost))

            if new_row[-1] <= max_distance and node.is_terminal:
                for n in node.nodes:
                    dist = new_row[-1]
                    if n.id not in results or dist < results[n.id][1]:
                        results[n.id] = (n, dist)

            if min(new_row) <= max_distance:
                for next_char, next_node in node.children.items():
                    _search_recursive(
                        next_node, next_char, new_row, current_str + next_char
                    )

        for first_char, child_node in self.root.children.items():
            _search_recursive(child_node, first_char, current_row, first_char)

        sorted_results = sorted(results.values(), key=lambda x: x[1])
        return sorted_results


# -----------------------------------------------------------------------------
# In-Memory Phonetic Gazetteer Matcher
# -----------------------------------------------------------------------------
class PhoneticGazetteerMatcher:
    """
    Indexes all BangladeshGazetteer entities into:
      1. Primary phonetic inverted index
      2. Secondary consonant skeleton inverted index
      3. Prefix Phonetic Trie
    """

    def __init__(self, gazetteer: Optional[BangladeshGazetteer] = None) -> None:
        self.gazetteer = gazetteer if gazetteer is not None else BangladeshGazetteer()
        self.trie = PhoneticTrie()
        self.primary_index: Dict[str, List[AdminNode]] = {}
        self.skeleton_index: Dict[str, List[AdminNode]] = {}
        self.name_to_nodes: Dict[str, List[AdminNode]] = {}
        self._index_all_nodes()

    def _index_name(self, name: str, node: AdminNode) -> None:
        if not name or not name.strip():
            return
        cleaned = name.strip()
        cleaned_lower = cleaned.lower()

        # Direct name map
        if cleaned_lower not in self.name_to_nodes:
            self.name_to_nodes[cleaned_lower] = []
        if node not in self.name_to_nodes[cleaned_lower]:
            self.name_to_nodes[cleaned_lower].append(node)

        # Phonetic keys
        pk, sk = banglish_phonetic_key(cleaned)
        if pk:
            if pk not in self.primary_index:
                self.primary_index[pk] = []
            if node not in self.primary_index[pk]:
                self.primary_index[pk].append(node)
            self.trie.insert(pk, node)

        if sk:
            if sk not in self.skeleton_index:
                self.skeleton_index[sk] = []
            if node not in self.skeleton_index[sk]:
                self.skeleton_index[sk].append(node)

    def _index_all_nodes(self) -> None:
        for node in self.gazetteer.all_nodes:
            self._index_name(node.name, node)
            if node.bn_name:
                self._index_name(node.bn_name, node)

    def search(
        self, query: str, max_results: int = 10
    ) -> List[Tuple[AdminNode, float]]:
        """
        Searches gazetteer by query string using exact, phonetic, and Levenshtein ranking.
        Returns list of (AdminNode, score) with score from 0.0 to 1.0.
        """
        if not query or not query.strip():
            return []

        q_clean = query.strip()
        q_lower = q_clean.lower()
        pk, sk = banglish_phonetic_key(q_clean)

        candidates: Dict[str, Tuple[AdminNode, float]] = {}

        # 1. Exact string matches
        if q_lower in self.name_to_nodes:
            for node in self.name_to_nodes[q_lower]:
                candidates[node.id] = (node, 1.0)

        # 2. Exact primary phonetic matches
        if pk in self.primary_index:
            for node in self.primary_index[pk]:
                if node.id not in candidates:
                    # Score based on Levenshtein distance between original name and query
                    dist = min(
                        levenshtein_distance(q_lower, node.name.lower()),
                        levenshtein_distance(q_clean, node.bn_name)
                        if node.bn_name
                        else 99,
                    )
                    score = max(0.6, 1.0 - (dist * 0.1))
                    candidates[node.id] = (node, score)

        # 3. Consonant skeleton matches (handles vowel variations like e vs i, o vs u)
        if sk in self.skeleton_index:
            for node in self.skeleton_index[sk]:
                if node.id not in candidates:
                    dist = min(
                        levenshtein_distance(q_lower, node.name.lower()),
                        levenshtein_distance(q_clean, node.bn_name)
                        if node.bn_name
                        else 99,
                    )
                    score = max(0.5, 0.9 - (dist * 0.1))
                    candidates[node.id] = (node, score)


        # 3. Trie prefix / fuzzy matches
        if len(candidates) < max_results:
            prefix_matches = self.trie.search_prefix(pk)
            for node in prefix_matches:
                if node.id not in candidates:
                    dist = levenshtein_distance(q_lower, node.name.lower())
                    score = max(0.4, 0.85 - (dist * 0.1))
                    candidates[node.id] = (node, score)

        # 4. Fuzzy Levenshtein on Trie
        if len(candidates) < max_results:
            fuzzy_matches = self.trie.fuzzy_search(pk, max_distance=1)
            for node, dist in fuzzy_matches:
                if node.id not in candidates:
                    score = max(0.3, 0.7 - (dist * 0.15))
                    candidates[node.id] = (node, score)

        # Sort by score descending
        sorted_candidates = sorted(
            candidates.values(), key=lambda x: x[1], reverse=True
        )
        return sorted_candidates[:max_results]

    def MeasureCollisionRate(self) -> Dict[str, Any]:
        """
        Measures and prints exact phonetic collision statistics across the entire gazetteer.
        Returns a dictionary with comprehensive statistics.
        """
        total_entities = len(self.gazetteer.all_nodes)
        total_name_keys = sum(len(v) for v in self.primary_index.values())
        unique_keys = len(self.primary_index)

        # Collisions: keys mapped to > 1 distinct administrative node
        collision_keys = {
            k: v for k, v in self.primary_index.items() if len(v) > 1
        }
        total_collisions = len(collision_keys)
        max_collision_key, max_collision_nodes = max(
            self.primary_index.items(), key=lambda x: len(x[1])
        )

        avg_collisions_per_key = (
            total_name_keys / unique_keys if unique_keys > 0 else 0.0
        )
        collision_rate_pct = (
            (total_collisions / unique_keys) * 100.0 if unique_keys > 0 else 0.0
        )

        # Top 10 collision keys
        top_collisions = sorted(
            self.primary_index.items(), key=lambda x: len(x[1]), reverse=True
        )[:10]

        top_collision_info = [
            {
                "key": k,
                "node_count": len(nodes),
                "sample_nodes": [
                    f"{n.name} ({n.district.name if n.district else 'None'})"
                    for n in nodes[:4]
                ],
            }
            for k, nodes in top_collisions
        ]

        # Formatted output
        print("=" * 68)
        print("   BANGLADESH GAZETTEER PHONETIC COLLISION RATE REPORT")
        print("=" * 68)
        print(f" Total Administrative Nodes: {total_entities}")
        print(f" Unique Phonetic Primary Keys: {unique_keys}")
        print(f" Average Entities Per Key: {avg_collisions_per_key:.2f}")
        print(f" Keys with Collisions (>1 node): {total_collisions} ({collision_rate_pct:.1f}%)")
        print(
            f" Highest Collision Key: '{max_collision_key}' with {len(max_collision_nodes)} nodes"
        )
        print("-" * 68)
        print(" Top 10 Phonetic Collision Clusters:")
        for idx, item in enumerate(top_collision_info, 1):
            samples = ", ".join(item["sample_nodes"])
            # Format safely for Windows cp1252 console
            safe_samples = samples.encode("ascii", "backslashreplace").decode("ascii")
            print(f"  {idx:2d}. {item['key']:<12} -> {item['node_count']:2d} nodes [{safe_samples}]")
        print("=" * 68)


        return {
            "total_entities": total_entities,
            "unique_primary_keys": unique_keys,
            "total_collisions": total_collisions,
            "collision_rate_pct": collision_rate_pct,
            "avg_collisions_per_key": avg_collisions_per_key,
            "max_collision_key": max_collision_key,
            "max_collision_count": len(max_collision_nodes),
            "top_collisions": top_collision_info,
        }
