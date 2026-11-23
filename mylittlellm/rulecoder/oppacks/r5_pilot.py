"""Round 5 pilot operators: named definitions, formulas, conventions and standard routines, all gated.

The bank behind this pack is invented practice exercises, and the pack is judged on real tasks phrased by
other people, so every rule is a library entry under its usual names: a named number or digit property, a
cipher or text convention, a calendar rule, a classic list scan, a matrix, grid, tree or graph routine.
Nothing here is ungated: each ungated rule adds accidental matches to every task, and these rules are only
meaningful when the description names them. As in the round-3 packs, the search function of every operator
is compiled from its emitted template, so the program the search verifies is the program that is shipped.
"""
import datetime
import importlib
import math
import re

# Modules the engine imports automatically in generated answers; templates may use them freely.
MODULES = ("bisect", "collections", "datetime", "fractions", "functools", "heapq", "itertools", "math",
           "operator", "re", "statistics", "string")

# ---------------------------------------------------------------- helper sources
# Pasted into generated answers. The engine compiles helpers with `math` and `functools` only, so any other
# module is imported inside the function. None of them modifies its arguments.

PERSISTENCE = '''def _r5_persistence(n, multiply):
    # Steps of replacing n by the product (or the sum) of its decimal digits until one digit is left.
    steps = 0
    while n >= 10:
        digits = [int(d) for d in str(n)]
        n = math.prod(digits) if multiply else sum(digits)
        steps += 1
    return steps
'''

COLLATZ = '''def _r5_collatz(n):
    # The 3n + 1 (hailstone) sequence from n down to 1, both ends included.
    sequence = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        sequence.append(n)
    return sequence
'''

LOOK_AND_SAY = '''def _r5_look_and_say(n):
    # Term n (counted from 1) of the sequence that starts with '1' and reads off the runs of the term before.
    import itertools
    term = '1'
    for _ in range(n - 1):
        term = ''.join(str(len(list(group))) + digit for digit, group in itertools.groupby(term))
    return term
'''

TOTIENT = '''def _r5_totient(n):
    # Product formula over the distinct prime factors: phi(n) = n * prod(1 - 1 / p).
    result, rest, p = n, n, 2
    while p * p <= rest:
        if rest % p == 0:
            while rest % p == 0:
                rest //= p
            result -= result // p
        p += 1
    if rest > 1:
        result -= result // rest
    return result
'''

PARTITIONS = '''def _r5_partition_count(n):
    # Ways to write n as a sum of positive integers, order ignored: allow one more part size at a time.
    ways = [1] + [0] * n
    for part in range(1, n + 1):
        for total in range(part, n + 1):
            ways[total] += ways[total - part]
    return ways[n]
'''

NEXT_PRIME = '''def _r5_next_prime(n):
    # Trial division is enough: prime gaps are short, so few candidates are tested.
    candidate = max(n + 1, 2)
    while any(candidate % d == 0 for d in range(2, math.isqrt(candidate) + 1)):
        candidate += 1
    return candidate
'''

TWIN_PRIMES = '''def _r5_twin_primes(n):
    # Pairs (p, p + 2) of primes with p + 2 <= n, read off one sieve of Eratosthenes.
    flags = [False, False] + [True] * max(n - 1, 0)
    for p in range(2, math.isqrt(max(n, 0)) + 1):
        if flags[p]:
            flags[p * p::p] = [False] * len(flags[p * p::p])
    return [(p, p + 2) for p in range(2, n - 1) if flags[p] and flags[p + 2]]
'''

GOLDBACH = '''def _r5_goldbach_pair(n):
    # The primes p <= q with p + q == n and the smallest p; None when there is no such pair.
    def is_prime(k):
        return k > 1 and all(k % d for d in range(2, math.isqrt(k) + 1))
    for p in range(2, n // 2 + 1):
        if is_prime(p) and is_prime(n - p):
            return (p, n - p)
    return None
'''

EMIRP = '''def _r5_is_emirp(n):
    # A prime whose digit reversal is a different prime (13 and 31); palindromic primes do not count.
    def is_prime(k):
        return k > 1 and all(k % d for d in range(2, math.isqrt(k) + 1))
    reverse = int(str(n)[::-1])
    return reverse != n and is_prime(n) and is_prime(reverse)
'''

ALIQUOT = '''def _r5_aliquot_sum(n):
    # Sum of the divisors smaller than n, pairing d with n // d so that large n stay cheap.
    total = 0
    for d in range(1, math.isqrt(n) + 1):
        if n % d == 0:
            total += d if d * d == n else d + n // d
    return total - n
'''

KAPREKAR_ROUTINE = '''def _r5_kaprekar_iterations(n):
    # Steps of "digits in descending order minus digits in ascending order", four digits kept, until 6174.
    steps = 0
    while n != 6174:
        digits = '%04d' % n
        n = int(''.join(sorted(digits, reverse=True))) - int(''.join(sorted(digits)))
        steps += 1
    return steps
'''

KAPREKAR_NUMBER = '''def _r5_is_kaprekar(n):
    # The square split into its last d digits (d = digits of n) and the rest adds up to n; the right part > 0.
    square, d = str(n * n), len(str(n))
    right = int(square[-d:])
    left = int(square[:-d]) if len(square) > d else 0
    return right > 0 and left + right == n
'''

MULTIPLICATIVE_ORDER = '''def _r5_multiplicative_order(a, n):
    # Smallest k > 0 with a ** k % n == 1; it exists because a and n are coprime.
    k, value = 1, a % n
    while value != 1:
        value = value * a % n
        k += 1
    return k
'''

NUMBER_WORDS = '''def _r5_number_words(n):
    # American style: hyphenated tens ('forty-two') and no 'and' after the hundreds.
    ones = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
            'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

    def below_thousand(k):
        words = []
        if k >= 100:
            words.append(ones[k // 100] + ' hundred')
            k %= 100
        if k >= 20:
            words.append(tens[k // 10] + ('-' + ones[k % 10] if k % 10 else ''))
        elif k or not words:
            words.append(ones[k])
        return ' '.join(words)

    if n == 0:
        return 'zero'
    parts = []
    for value, name in ((10 ** 9, ' billion'), (10 ** 6, ' million'), (1000, ' thousand'), (1, '')):
        if n >= value:
            parts.append(below_thousand(n // value) + name)
            n %= value
    return ' '.join(parts)
'''

COLUMN_LABEL = '''def _r5_column_label(n):
    # Bijective base 26, as spreadsheet columns are named: 1 is A, 26 is Z, 27 is AA.
    label = ''
    while n > 0:
        n, rest = divmod(n - 1, 26)
        label = chr(65 + rest) + label
    return label
'''

VIGENERE = '''def _r5_vigenere(text, key, direction=1):
    # Letters keep their case; any other character is copied and does not use up a key letter.
    shifts = [ord(c) - ord('a') for c in key.lower()]
    out, used = [], 0
    for ch in text:
        if 'a' <= ch.lower() <= 'z':
            base = ord('A') if ch.isupper() else ord('a')
            out.append(chr((ord(ch) - base + direction * shifts[used % len(shifts)]) % 26 + base))
            used += 1
        else:
            out.append(ch)
    return ''.join(out)
'''

MORSE_CODES = ("{'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',\n"
               "             'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',\n"
               "             'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-',\n"
               "             'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----',\n"
               "             '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',\n"
               "             '8': '---..', '9': '----.'}")

MORSE_ENCODE = '''def _r5_morse_encode(text):
    # International Morse code: letters of a word separated by one space, words by ' / '.
    codes = %s
    return ' / '.join(' '.join(codes[c] for c in word) for word in text.upper().split())
''' % MORSE_CODES

MORSE_DECODE = '''def _r5_morse_decode(code):
    # Codes separated by spaces, words by '/'; the text comes back in capitals.
    codes = %s
    letters = {value: key for key, value in codes.items()}
    return ' '.join(''.join(letters[c] for c in word.split()) for word in code.split('/') if word.strip())
''' % MORSE_CODES

PIG_LATIN = '''def _r5_pig_latin(word):
    # A leading vowel takes 'way'; otherwise the consonants before the first vowel move behind it, then 'ay'.
    for i, ch in enumerate(word):
        if ch.lower() in 'aeiou':
            return word + 'way' if i == 0 else word[i:] + word[:i] + 'ay'
    return word + 'ay'
'''

WORD_WRAP = '''def _r5_word_wrap(text, width):
    # Greedy filling with whole words; a word longer than the width gets a line of its own.
    lines, current = [], ''
    for word in text.split():
        if current and len(current) + 1 + len(word) <= width:
            current += ' ' + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
'''

ALTERNATE_CASE = '''def _r5_alternate_case(text, letters_only):
    # Upper, lower, upper ... counted over the letters only, or over every position.
    out, upper = [], True
    for ch in text:
        if ch.isalpha() or not letters_only:
            out.append(ch.upper() if upper else ch.lower())
            upper = not upper
        else:
            out.append(ch)
    return ''.join(out)
'''

RAIL_FENCE = '''def _r5_rail_fence(text, rails):
    # The characters zigzag down and up across the rails, which are then read from top to bottom.
    if rails <= 1:
        return text
    rows, row, step = [''] * rails, 0, 1
    for ch in text:
        rows[row] += ch
        if row == 0:
            step = 1
        elif row == rails - 1:
            step = -1
        row += step
    return ''.join(rows)
'''

KEYWORD_CIPHER = '''def _r5_keyword_cipher(text, keyword):
    # Cipher alphabet: the distinct letters of the keyword, then the rest of the alphabet in order.
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    key = ''.join(dict.fromkeys(keyword.lower() + alphabet))
    return text.translate(str.maketrans(alphabet + alphabet.upper(), key + key.upper()))
'''

LONGEST_PALINDROME = '''def _r5_longest_palindrome(text):
    # Expand around every centre; only a strictly longer palindrome replaces the best, so ties keep the leftmost.
    best = text[:1]
    for centre in range(2 * len(text) - 1):
        left, right = centre // 2, (centre + 1) // 2
        while left >= 0 and right < len(text) and text[left] == text[right]:
            left, right = left - 1, right + 1
        if right - left - 1 > len(best):
            best = text[left + 1:right]
    return best
'''

LONGEST_DISTINCT_RUN = '''def _r5_longest_distinct_run(items):
    # Sliding window: it restarts just after the earlier copy of the current item.
    best, start, last = 0, 0, {}
    for i, item in enumerate(items):
        if last.get(item, -1) >= start:
            start = last[item] + 1
        last[item] = i
        best = max(best, i - start + 1)
    return best
'''

LONGEST_COMMON_SUBSTRING = '''def _r5_longest_common_substring(a, b):
    # Lengths of common suffixes; the first longest one found ends, and so starts, earliest in a.
    best_length, best_end = 0, 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best_length:
                    best_length, best_end = current[j], i
        previous = current
    return a[best_end - best_length:best_end]
'''

SIMPLIFY_PATH = '''def _r5_simplify_path(path):
    # Empty and '.' segments vanish, '..' removes the directory before it (never above the root).
    parts = []
    for part in path.split('/'):
        if part == '..':
            if parts:
                parts.pop()
        elif part and part != '.':
            parts.append(part)
    return '/' + '/'.join(parts)
'''

MIN_ADD_PARENTHESES = '''def _r5_min_add_to_balance(text):
    # An unmatched ')' needs a '(' inserted before it; every '(' still open at the end needs a ')'.
    opened = added = 0
    for ch in text:
        if ch == '(':
            opened += 1
        elif ch == ')':
            if opened:
                opened -= 1
            else:
                added += 1
    return added + opened
'''

EDIT_DISTANCE = '''def _r5_edit_distance(a, b):
    # Levenshtein distance, one row of the table at a time.
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]
'''

ONE_EDIT = '''def _r5_one_edit_apart(a, b):
    # Exactly one insertion, deletion or replacement turns one string into the other.
    if len(a) > len(b):
        a, b = b, a
    if len(b) - len(a) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    i = next((k for k in range(len(a)) if a[k] != b[k]), len(a))
    return a[i:] == b[i + 1:]
'''

INTERLEAVING = '''def _r5_is_interleaving(a, b, c):
    # reach[j]: the first i + j characters of c can be made from a[:i] and b[:j], keeping both orders.
    if len(a) + len(b) != len(c):
        return False
    reach = [True] + [False] * len(b)
    for j in range(1, len(b) + 1):
        reach[j] = reach[j - 1] and b[j - 1] == c[j - 1]
    for i in range(1, len(a) + 1):
        reach[0] = reach[0] and a[i - 1] == c[i - 1]
        for j in range(1, len(b) + 1):
            reach[j] = (reach[j] and a[i - 1] == c[i + j - 1]) or (reach[j - 1] and b[j - 1] == c[i + j - 1])
    return reach[len(b)]
'''

WILDCARD = '''def _r5_wildcard_match(text, pattern):
    # '?' matches one character and '*' any run of characters, the empty one included; all of text must match.
    reach = [True] + [False] * len(pattern)
    for j, p in enumerate(pattern, 1):
        reach[j] = reach[j - 1] and p == '*'
    for ch in text:
        diagonal, reach[0] = reach[0], False
        for j, p in enumerate(pattern, 1):
            above = reach[j]
            reach[j] = (above or reach[j - 1]) if p == '*' else (diagonal and (p == '?' or p == ch))
            diagonal = above
    return reach[len(pattern)]
'''

LUHN = '''def _r5_luhn(number):
    # Spaces are ignored; from the right every second digit is doubled (minus 9 above 9); the total ends in 0.
    digits = number.replace(' ', '')
    if len(digits) < 2 or any(c not in '0123456789' for c in digits):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0
'''

ISBN10 = '''def _r5_isbn10_valid(code):
    # Hyphens and spaces are layout; the weighted sum 10*d1 + 9*d2 + ... + 1*d10 must be divisible by 11.
    s = code.replace('-', '').replace(' ', '')
    if len(s) != 10 or any(c not in '0123456789' for c in s[:9]) or s[9] not in '0123456789Xx':
        return False
    total = sum((10 - i) * int(c) for i, c in enumerate(s[:9])) + (10 if s[9] in 'Xx' else int(s[9]))
    return total % 11 == 0
'''

MOUNTAIN = '''def _r5_is_mountain(items):
    # Strictly up to a single peak, then strictly down, with at least one element on each side of the peak.
    n, i = len(items), 0
    while i + 1 < n and items[i] < items[i + 1]:
        i += 1
    if i == 0 or i == n - 1:
        return False
    while i + 1 < n and items[i] > items[i + 1]:
        i += 1
    return i == n - 1
'''

INCREASING_TRIPLET = '''def _r5_has_increasing_triplet(items):
    # Keep the smallest value seen and the smallest value that has something smaller before it.
    first = second = None
    for value in items:
        if first is None or value <= first:
            first = value
        elif second is None or value <= second:
            second = value
        else:
            return True
    return False
'''

SUDOKU = '''def _r5_is_sudoku(grid):
    # A complete n x n grid, n a perfect square: every row, column and box holds 1..n exactly once.
    n = len(grid)
    box = math.isqrt(n)
    target = set(range(1, n + 1))
    groups = [list(row) for row in grid] + [list(col) for col in zip(*grid)]
    groups += [[grid[r][c] for r in range(top, top + box) for c in range(left, left + box)]
               for top in range(0, n, box) for left in range(0, n, box)]
    return all(len(group) == n and set(group) == target for group in groups)
'''

LIFE = '''def _r5_life_step(grid):
    # Cells outside the grid are dead; a live cell with 2 or 3 live neighbours survives, a dead one with 3 is born.
    rows, cols = len(grid), len(grid[0])

    def neighbours(r, c):
        return sum(grid[i][j] for i in range(max(r - 1, 0), min(r + 2, rows))
                   for j in range(max(c - 1, 0), min(c + 2, cols)) if (i, j) != (r, c))
    return [[1 if neighbours(r, c) == 3 or (grid[r][c] and neighbours(r, c) == 2) else 0 for c in range(cols)]
            for r in range(rows)]
'''

ISLANDS = '''def _r5_islands(grid):
    # Land is 1 (or '1'); land cells that touch horizontally or vertically belong to one island.
    land = {(r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] in (1, '1')}
    count = 0
    while land:
        count += 1
        stack = [land.pop()]
        while stack:
            r, c = stack.pop()
            for cell in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if cell in land:
                    land.remove(cell)
                    stack.append(cell)
    return count
'''

ISLAND_PERIMETER = '''def _r5_island_perimeter(grid):
    # Every side of a land cell that faces water or the edge of the grid is part of the coastline.
    def land(r, c):
        return 0 <= r < len(grid) and 0 <= c < len(grid[r]) and grid[r][c] in (1, '1')
    return sum(not land(r + dr, c + dc) for r in range(len(grid)) for c in range(len(grid[r])) if land(r, c)
               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
'''

SPIRAL = '''def _r5_spiral_order(matrix):
    # Take the top row, turn the rest a quarter counter-clockwise, repeat: that walks the border clockwise.
    rows, out = [list(row) for row in matrix], []
    while rows:
        out.extend(rows.pop(0))
        rows = [list(col) for col in zip(*rows)][::-1]
    return out
'''

DETERMINANT = '''def _r5_determinant(matrix):
    # Bareiss elimination: on integers every division is exact, so the result is exact; rows swap at zero pivots.
    m = [list(row) for row in matrix]
    n, sign, previous = len(m), 1, 1
    exact = all(isinstance(v, int) for row in m for v in row)
    for k in range(n - 1):
        if m[k][k] == 0:
            swap = next((i for i in range(k + 1, n) if m[i][k] != 0), None)
            if swap is None:
                return 0
            m[k], m[swap] = m[swap], m[k]
            sign = -sign
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                value = m[i][j] * m[k][k] - m[i][k] * m[k][j]
                m[i][j] = value // previous if exact else value / previous
        previous = m[k][k]
    return sign * m[-1][-1]
'''

MERGE_INTERVALS = '''def _r5_merge_intervals(intervals):
    # Sorted by start, an interval that starts no later than the current one ends extends it.
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    kind = type(intervals[0]) if intervals else tuple
    return type(intervals)(kind(pair) for pair in merged)
'''

KNAPSACK = '''def _r5_knapsack(items, capacity):
    # 0/1 knapsack over (weight, value) items; capacities are scanned downwards so each item is used once.
    best = [0] * (capacity + 1)
    for weight, value in items:
        for room in range(capacity, weight - 1, -1):
            best[room] = max(best[room], best[room - weight] + value)
    return best[capacity]
'''

GROUP_BY_FIRST = '''def _r5_group_by_first(words, lower):
    # Words in their original order under their first character (lower-cased if asked).
    groups = {}
    for word in words:
        groups.setdefault(word[0].lower() if lower else word[0], []).append(word)
    return groups
'''

FLATTEN_DICT = '''def _r5_flatten_dict(data, separator='.', prefix=''):
    # Keys along the path are joined; an empty inner dictionary leaves nothing behind.
    flat = {}
    for key, value in data.items():
        path = prefix + separator + str(key) if prefix else key
        if isinstance(value, dict):
            flat.update(_r5_flatten_dict(value, separator, str(path)))
        else:
            flat[path] = value
    return flat
'''

GET_PATH = '''def _r5_get_path(data, keys):
    # Follow the keys through nested dictionaries; None as soon as one is missing.
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data
'''

DEEP_MERGE = '''def _r5_deep_merge(base, override):
    # Values of the second dictionary win, except that two dictionaries under one key are merged in turn.
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _r5_deep_merge(result[key], value)
        else:
            result[key] = value
    return result
'''

DEEP_REVERSE = '''def _r5_deep_reverse(items):
    return type(items)(_r5_deep_reverse(v) if isinstance(v, (list, tuple)) else v for v in reversed(items))
'''

SWAP_LEVELS = '''def _r5_swap_levels(data):
    # d[a][b] becomes result[b][a]; inner keys appear in the order in which they are first met.
    result = {}
    for outer, inner in data.items():
        for key, value in inner.items():
            result.setdefault(key, {})[outer] = value
    return result
'''

UNDIRECTED = '''def _r5_undirected_adjacency(edges):
    # Every edge is entered in both directions; neighbour lists are sorted, so set order never shows.
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    return {vertex: sorted(neighbours) for vertex, neighbours in adjacency.items()}
'''

BFS = '''def _r5_bfs_order(graph, start):
    # Breadth first; neighbours are queued in the order in which they are listed, each vertex once.
    import collections
    order, seen, queue = [start], {start}, collections.deque([start])
    while queue:
        vertex = queue.popleft()
        for neighbour in graph.get(vertex, ()):
            if neighbour not in seen:
                seen.add(neighbour)
                order.append(neighbour)
                queue.append(neighbour)
    return order
'''

DFS = '''def _r5_dfs_order(graph, start):
    # Recursive depth first: a vertex, then everything reachable from its first unvisited neighbour, and so on.
    order, seen = [], set()

    def visit(vertex):
        seen.add(vertex)
        order.append(vertex)
        for neighbour in graph.get(vertex, ()):
            if neighbour not in seen:
                visit(neighbour)
    visit(start)
    return order
'''

TOPOLOGICAL = '''def _r5_topological_order(dependencies):
    # Kahn's algorithm on "task -> tasks it needs"; the smallest ready task goes first, so the order is unique.
    import heapq
    needs = {task: set(before) for task, before in dependencies.items()}
    for before in dependencies.values():
        for task in before:
            needs.setdefault(task, set())
    users = {task: [] for task in needs}
    for task, before in needs.items():
        for b in before:
            users[b].append(task)
    ready = [task for task, before in needs.items() if not before]
    heapq.heapify(ready)
    order = []
    while ready:
        task = heapq.heappop(ready)
        order.append(task)
        for user in users[task]:
            needs[user].discard(task)
            if not needs[user]:
                heapq.heappush(ready, user)
    if len(order) != len(needs):
        raise ValueError('the dependencies contain a cycle')
    return order
'''

# The edge form ("node -> nodes that come after it") is turned into the dependency form first.
TOPOLOGICAL_EDGES = '''def _r5_topological_order_edges(graph):
    dependencies = {node: [] for node in graph}
    for node, successors in graph.items():
        for successor in successors:
            dependencies.setdefault(successor, []).append(node)
    return _r5_topological_order(dependencies)


''' + TOPOLOGICAL

DIJKSTRA = '''def _r5_shortest_distances(graph, source):
    # Dijkstra with a heap; only reachable vertices appear. The counter keeps vertices from being compared.
    import heapq
    distances, heap, done, count = {source: 0}, [(0, 0, source)], set(), 1
    while heap:
        distance, _, vertex = heapq.heappop(heap)
        if vertex in done:
            continue
        done.add(vertex)
        edges = graph.get(vertex, {})
        for neighbour, weight in (edges.items() if isinstance(edges, dict) else edges):
            candidate = distance + weight
            if neighbour not in distances or candidate < distances[neighbour]:
                distances[neighbour] = candidate
                heapq.heappush(heap, (candidate, count, neighbour))
                count += 1
    return distances
'''

ALL_PATHS = '''def _r5_all_paths(graph, start, end):
    # Depth first in the listed order of neighbours; the graph is acyclic, so every path ends.
    if start == end:
        return [[start]]
    return [[start] + rest for neighbour in graph.get(start, ()) for rest in _r5_all_paths(graph, neighbour, end)]
'''

BIPARTITE = '''def _r5_is_bipartite(graph):
    # Two-colour every component; an edge between two vertices of one colour breaks it.
    vertices = range(len(graph)) if isinstance(graph, list) else graph
    colour = {}
    for start in vertices:
        if start in colour:
            continue
        colour[start] = 0
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbour in (graph[vertex] if isinstance(graph, list) else graph.get(vertex, ())):
                if neighbour not in colour:
                    colour[neighbour] = 1 - colour[vertex]
                    stack.append(neighbour)
                elif colour[neighbour] == colour[vertex]:
                    return False
    return True
'''

DIRECTED_CYCLE = '''def _r5_has_directed_cycle(graph):
    # Depth first; reaching a vertex that is still on the current path closes a cycle (a self-loop included).
    state = {}

    def visit(vertex):
        state[vertex] = 1
        for neighbour in graph.get(vertex, ()):
            if state.get(neighbour) == 1 or (neighbour not in state and visit(neighbour)):
                return True
        state[vertex] = 2
        return False
    return any(vertex not in state and visit(vertex) for vertex in graph)
'''

IS_BST = '''def _r5_is_bst(tree, low=None, high=None):
    # Trees are (value, left, right) with None for a missing child; every value lies strictly inside the
    # bounds set by its ancestors.
    if tree is None:
        return True
    value, left, right = tree
    if (low is not None and value <= low) or (high is not None and value >= high):
        return False
    return _r5_is_bst(left, low, value) and _r5_is_bst(right, value, high)
'''

TREE_ORDER = '''def _r5_tree_order(tree, order):
    # Trees are (value, left, right) with None for a missing child.
    if tree is None:
        return []
    value, left, right = tree
    left, right = _r5_tree_order(left, order), _r5_tree_order(right, order)
    if order == 'pre':
        return [value] + left + right
    if order == 'post':
        return left + right + [value]
    return left + [value] + right
'''

ROOT_PATH = '''def _r5_path_from_root(parents, node):
    # Follow the parent links up to the root (whose parent is None), then read the chain backwards.
    path = []
    while node is not None:
        path.append(node)
        node = parents[node]
    return path[::-1]
'''

TICTACTOE = '''def _r5_tictactoe_winner(board):
    # Rows, columns and both diagonals of a 3 x 3 board; None when nobody has three in a line.
    cells = [list(row) for row in board]
    lines = cells + [list(col) for col in zip(*cells)]
    lines += [[cells[i][i] for i in range(3)], [cells[i][2 - i] for i in range(3)]]
    for player in ('X', 'O', 'x', 'o'):
        if [player] * 3 in lines:
            return player
    return None
'''

MINESWEEPER = '''def _r5_minesweeper(board):
    # Every cell that is not a mine ('*') becomes the number of mines among its up to eight neighbours.
    out = []
    for r in range(len(board)):
        line = ''
        for c in range(len(board[r])):
            if board[r][c] == '*':
                line += '*'
            else:
                line += str(sum(board[i][j] == '*' for i in range(max(r - 1, 0), min(r + 2, len(board)))
                                for j in range(max(c - 1, 0), min(c + 2, len(board[i])))))
        out.append(line)
    return out
'''

GRID_PATH = '''def _r5_grid_shortest_path(grid):
    # Breadth first over open cells ('.' or 0) from the top-left to the bottom-right corner; -1 if cut off.
    import collections
    rows, cols = len(grid), len(grid[0])

    def is_open(r, c):
        return grid[r][c] in ('.', 0)
    if not is_open(0, 0) or not is_open(rows - 1, cols - 1):
        return -1
    distance = {(0, 0): 0}
    queue = collections.deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        if (r, c) == (rows - 1, cols - 1):
            return distance[(r, c)]
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in distance and is_open(nr, nc):
                distance[(nr, nc)] = distance[(r, c)] + 1
                queue.append((nr, nc))
    return -1
'''

OBSTACLE_PATHS = '''def _r5_paths_avoiding(grid):
    # Paths that move right or down only and never enter a cell holding 1, one row of counts at a time.
    ways = [1] + [0] * (len(grid[0]) - 1)
    for row in grid:
        for c in range(len(row)):
            if row[c] == 1:
                ways[c] = 0
            elif c > 0:
                ways[c] += ways[c - 1]
    return ways[-1]
'''

GROUP_ANAGRAMS = '''def _r5_group_anagrams(words):
    # Groups in the order of their first word, words in their original order.
    groups = {}
    for word in words:
        groups.setdefault(''.join(sorted(word)), []).append(word)
    return list(groups.values())
'''

GREEDY_CHANGE = '''def _r5_greedy_change(amount, denominations):
    # Largest denomination first, as many of each as fit; unused denominations are left out.
    counts = {}
    for coin in sorted(denominations, reverse=True):
        number, amount = divmod(amount, coin)
        if number:
            counts[coin] = number
    return counts
'''

QUERY_STRING = '''def _r5_parse_query(query):
    # 'a=1&b=2' gives {'a': '1', 'b': '2'}; a key given twice keeps its last value.
    result = {}
    for pair in (query.split('&') if query else []):
        key, _, value = pair.partition('=')
        result[key] = value
    return result
'''

JOSEPHUS = '''def _r5_josephus_order(n, k):
    # People 1..n in a circle; counting starts at person 1 and every k-th person leaves.
    people, order, i = list(range(1, n + 1)), [], 0
    while people:
        i = (i + k - 1) % len(people)
        order.append(people.pop(i))
    return order
'''

NEXT_GREATER = '''def _r5_next_greater(items, distance):
    # One stack scan: each position waits on the stack until a strictly greater value arrives.
    result = [0 if distance else -1] * len(items)
    waiting = []
    for i, value in enumerate(items):
        while waiting and items[waiting[-1]] < value:
            j = waiting.pop()
            result[j] = i - j if distance else value
        waiting.append(i)
    return result
'''

TRAPPED_WATER = '''def _r5_trapped_water(heights):
    # Water above a bar reaches the lower of the highest bars on its left and on its right.
    left, best = [], 0
    for h in heights:
        best = max(best, h)
        left.append(best)
    total, best = 0, 0
    for i in range(len(heights) - 1, -1, -1):
        best = max(best, heights[i])
        total += min(left[i], best) - heights[i]
    return total
'''

SPLIT_ON = '''def _r5_split_on(items, separator):
    # Like str.split for lists: k separators always give k + 1 parts, empty ones included.
    parts = [[]]
    for item in items:
        if item == separator:
            parts.append([])
        else:
            parts[-1].append(item)
    return parts
'''

SUBARRAY_SUM_COUNT = '''def _r5_count_subarray_sums(items, target):
    # A run i..j adds up to target when prefix(j) - prefix(i - 1) == target: count the earlier prefixes.
    seen, total, count = {0: 1}, 0, 0
    for value in items:
        total += value
        count += seen.get(total - target, 0)
        seen[total] = seen.get(total, 0) + 1
    return count
'''

KTH_PERMUTATION = '''def _r5_kth_permutation(n, k):
    # Factorial number system: every block of (size - 1)! permutations fixes the next element.
    numbers, k, result = list(range(1, n + 1)), k - 1, []
    for size in range(n, 0, -1):
        index, k = divmod(k, math.factorial(size - 1))
        result.append(numbers.pop(index))
    return result
'''

PARTITION_DIFFERENCE = '''def _r5_min_partition_difference(items):
    # Subset sums reachable from the items; the best split puts one group's sum closest to half the total.
    total, reachable = sum(items), {0}
    for value in items:
        reachable |= {r + value for r in reachable}
    return min(abs(total - 2 * r) for r in reachable)
'''

INCREASING_RUN = '''def _r5_longest_increasing_run(items):
    # Longest stretch of neighbours in which every element is strictly greater than the one before.
    best = current = 1 if len(items) else 0
    for a, b in zip(items, items[1:]):
        current = current + 1 if b > a else 1
        best = max(best, current)
    return best
'''

CANCEL_PAIRS = '''def _r5_cancel_pairs(items):
    # A stack: an item equal to the top cancels it, which also removes the pairs that this exposes.
    stack = []
    for item in items:
        if stack and stack[-1] == item:
            stack.pop()
        else:
            stack.append(item)
    return stack
'''

RPN = '''def _r5_eval_rpn(tokens):
    # Integer arithmetic on a stack; division truncates toward zero, as in C and Java.
    stack = []
    for token in tokens:
        if token in ('+', '-', '*', '/'):
            b, a = stack.pop(), stack.pop()
            if token == '+':
                stack.append(a + b)
            elif token == '-':
                stack.append(a - b)
            elif token == '*':
                stack.append(a * b)
            else:
                quotient = abs(a) // abs(b)
                stack.append(quotient if (a >= 0) == (b > 0) else -quotient)
        else:
            stack.append(int(token))
    return stack[-1]
'''

SMALLEST_ARRANGEMENT = '''def _r5_smallest_arrangement(n):
    # Digits in ascending order, except that the smallest non-zero digit has to go in front of the zeros.
    digits = sorted(str(n))
    first = next((i for i, d in enumerate(digits) if d != '0'), 0)
    return int(digits[first] + ''.join(digits[:first] + digits[first + 1:]))
'''

STAIRS = '''def _r5_stair_ways(n, k):
    # Ordered ways to climb n steps taking 1 to k steps at a time: each count adds up the k counts before it.
    ways = [1]
    for _ in range(n):
        ways.append(sum(ways[-k:]))
    return ways[-1]
'''

BASE_DIGITS = '''def _r5_base_digits(n, base):
    # Digits of n in the given base, most significant first; zero has the single digit 0.
    digits = []
    while n:
        n, digit = divmod(n, base)
        digits.append(digit)
    return digits[::-1] or [0]
'''


REVERSE_ADD = '''def _r5_reverse_add_steps(n):
    # Add the digit reversal until a palindrome appears; a number that gets nowhere (196?) is given up on.
    steps = 0
    while str(n) != str(n)[::-1]:
        n += int(str(n)[::-1])
        steps += 1
        if steps > 500:
            raise ValueError('no palindrome within 500 reverse-and-add steps')
    return steps
'''

DECODE_WAYS = '''def _r5_decode_ways(digits):
    # Letters A..Z written as 1..26: a code ends at every position with one digit (not 0) or two (10..26).
    ways = [1] + [0] * len(digits)
    for i in range(1, len(digits) + 1):
        if digits[i - 1] != '0':
            ways[i] += ways[i - 1]
        if i > 1 and '10' <= digits[i - 2:i] <= '26':
            ways[i] += ways[i - 2]
    return ways[-1]
'''

MASK_DIGITS = '''def _r5_mask_digits(text, keep):
    # Every digit but the last `keep` digits becomes '*'; spaces and other characters stay where they are.
    total, seen, out = sum(c.isdigit() for c in text), 0, []
    for c in text:
        if c.isdigit():
            seen += 1
            out.append(c if seen > total - keep else '*')
        else:
            out.append(c)
    return ''.join(out)
'''

def register(lib):
    need, is_num, is_int, is_seq = lib.need, lib.is_num, lib.is_int, lib.is_seq
    scope = {name: importlib.import_module(name) for name in MODULES}
    sources = {}

    def add(arity, name, guard, template, keywords, helper=None, valid=None, kind="knowledge"):
        """Register a gated operator whose search function is its own emitted template behind a guard.

        The guard says where the rule applies and bounds the work; it never changes a value.
        """
        text = None
        if helper is not None:
            if helper not in sources:
                function, sources[helper] = lib.helper(helper)
                scope[function.__name__] = function
            text = sources[helper]
        params = ("_x", "_y", "_z")[:arity]
        body = eval("lambda " + ", ".join(params) + ": " + template.format(*params), scope)

        def run(*args):
            need(guard(*args))
            value = body(*args)
            need(valid is None or valid(value))
            return value
        {1: lib.unary, 2: lib.binary, 3: lib.ternary}[arity](
            "r5_" + name, run, template, keywords, gated=True, kind=kind, helper=text)

    def boolean(value):
        return value is True or value is False

    def pred(arity, name, guard, template, keywords, helper=None):
        # A predicate must answer a real bool: 0, 1 or None would never equal a True/False example anyway.
        add(arity, name, guard, template, keywords, helper, valid=boolean)

    def formula(arity, name, guard, template, keywords, helper=None):
        """A float formula plus its form rounded to a number of places that the description states.

        When the visible value happens to be exact (37.0, 25.0) the bare formula reproduces it and, being one
        step shorter, beats round(formula, k); the hidden tests then fail on the digits. The rounded form is one
        step too and is named by the formula's words together with 'round' or 'decimal', so the specificity
        tie-break picks it exactly when the description asks for rounding. Places run from 0 to 4 only: other
        numbers in such descriptions belong to the formula ("(F - 32) * 5 / 9"), and an exact value would be
        reproduced with any of them.
        """
        add(arity, name, guard, template, keywords, helper)
        rounded = "|".join(f"{alternative}&{word}" for alternative in keywords.split("|")
                           for word in ("round", "decimal"))
        add(arity + 1, name + "_rounded", lambda *args: guard(*args[:-1]) and is_int(args[-1]) and 0 <= args[-1] <= 4,
            "round(" + template + ", {" + str(arity) + "})", rounded, helper)

    def words(*groups):
        """Keywords that need one word of each group: words('a|b', 'x|y') is 'a&x|a&y|b&x|b&y'."""
        alternatives = [""]
        for group in groups:
            alternatives = [f"{a}&{w}" if a else w for a in alternatives for w in group.split("|")]
        return "|".join(alternatives)

    # ------------------------------------------------------------ guards
    # Sizes are bounded because the search calls an operator on every candidate value it builds.
    def text(x, low=0, high=600):
        return isinstance(x, str) and low <= len(x) <= high

    def natural(low, high):
        return lambda n: is_int(n) and low <= n <= high

    def seq(x, low=1, high=500):
        return is_seq(x) and low <= len(x) <= high

    def nums(x, low=1, high=500):
        return seq(x, low, high) and all(is_num(v) for v in x)

    def ints(x, low=1, high=500):
        return seq(x, low, high) and all(is_int(v) for v in x)

    def strs(x, low=1, high=500):
        return seq(x, low, high) and all(isinstance(v, str) for v in x)

    def scalar(v):
        return is_num(v) or isinstance(v, str)

    def flat(x, low=1, high=500):
        return seq(x, low, high) and all(scalar(v) for v in x)

    def comparable(x, low=1, high=500):
        return flat(x, low, high) and (all(is_num(v) for v in x) or all(isinstance(v, str) for v in x))

    def grid(x, cell=is_num, side=40):
        return (seq(x, 1, side) and all(is_seq(r) and 1 <= len(r) <= side for r in x)
                and len({len(r) for r in x}) == 1 and all(cell(v) for r in x for v in r))

    def square(x, cell=is_num, side=12):
        return grid(x, cell, side) and len(x) == len(x[0])

    def pairs(x, cell=is_num, low=1, high=300):
        return seq(x, low, high) and all(is_seq(p) and len(p) == 2 and cell(p[0]) and cell(p[1]) for p in x)

    def bit(v):
        return is_int(v) and v in (0, 1)

    def bit_grid(x):
        # Rows as lists of 0/1 (or '0'/'1'), or as strings of '0' and '1'.
        rows = [list(r) if isinstance(r, str) else r for r in x] if seq(x, 1, 50) else None
        return rows is not None and grid(rows, lambda v: bit(v) or v in ("0", "1"), 50)

    def adjacency(x, size=300):
        return (isinstance(x, dict) and 1 <= len(x) <= size and all(is_seq(v) for v in x.values())
                and sum(len(v) for v in x.values()) <= 3000)

    def number_dict(x):
        return isinstance(x, dict) and 1 <= len(x) <= 300 and all(is_num(v) for v in x.values())

    def records(x):
        return seq(x, 1, 300) and all(isinstance(r, dict) and r for r in x)

    def iso_date(x):
        if not (isinstance(x, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", x)):
            return False
        try:
            datetime.date.fromisoformat(x)
        except ValueError:
            return False
        return True

    def ymd(year, month, day):
        if not (is_int(year) and is_int(month) and is_int(day) and 1 <= year <= 9999):
            return False
        try:
            datetime.date(year, month, day)
        except ValueError:
            return False
        return True

    def ymd_tuple(x):
        return is_seq(x) and len(x) == 3 and ymd(*x)

    def tree(x, depth=0):
        return x is None or (depth < 60 and is_seq(x) and len(x) == 3 and scalar(x[0])
                             and tree(x[1], depth + 1) and tree(x[2], depth + 1))

    def real_tree(x):
        return x is not None and tree(x)

    ROUND = "rounded|rounding|round to|round off|round each|round all|round the|round it|round every"

    # ------------------------------------------------------------ digit and number definitions
    add(1, "digital_root", natural(0, 10 ** 30), "0 if {0} == 0 else 1 + ({0} - 1) % 9",
        "digital root|repeated digit sum|repeatedly&sum&digit|single digit&sum of&digit|single-digit&sum&digit")
    add(1, "multiplicative_persistence", natural(0, 10 ** 30), "_r5_persistence({0}, True)",
        "multiplicative persistence|persistence|digits&multiplied&single digit|product of&digits&single digit",
        PERSISTENCE)
    add(1, "additive_persistence", natural(0, 10 ** 30), "_r5_persistence({0}, False)",
        "additive persistence|persistence&sum|persistence&add", PERSISTENCE)
    collatz = "collatz|hailstone|3n+1|3n + 1|3 * n + 1|3*n+1|3x+1|3x + 1|3 * x + 1|syracuse"
    add(1, "collatz_steps", natural(1, 10 ** 6), "len(_r5_collatz({0})) - 1", collatz, COLLATZ)
    add(1, "collatz_sequence", natural(1, 10 ** 6), "_r5_collatz({0})", collatz, COLLATZ)
    add(1, "look_and_say", natural(1, 30), "_r5_look_and_say({0})",
        "look-and-say|look and say|count-and-say|count and say|look-say", LOOK_AND_SAY)
    add(1, "totient", natural(1, 10 ** 9), "_r5_totient({0})",
        "totient|phi function|euler's phi|euler phi|phi(n)|share no common factor|"
        + words("count|number of|how many", "coprime|co-prime|relatively prime"), TOTIENT)
    add(1, "factorial_trailing_zeros", natural(0, 10 ** 18), "sum({0} // 5 ** k for k in range(1, 40))",
        "factorial&trailing|factorial&zeros|factorial&zeroes")
    add(1, "double_factorial", natural(0, 500), "math.prod(range({0}, 0, -2))",
        "double factorial|semifactorial|semi-factorial|n!!")
    add(1, "partition_count", natural(0, 120), "_r5_partition_count({0})",
        "integer partition|partitions of|number of partitions|partition number|partition function|"
        "ways&sum of positive integers|ways of writing&sum|ways to write&sum|order of the terms does not matter",
        PARTITIONS)
    add(1, "next_prime", natural(-10 ** 6, 10 ** 7), "_r5_next_prime({0})",
        "next prime|" + words("smallest prime|first prime|least prime|nearest prime",
                              "greater|larger|bigger|after|exceed|above|following"), NEXT_PRIME)
    add(1, "twin_primes", natural(0, 10 ** 5), "_r5_twin_primes({0})", "twin prime", TWIN_PRIMES)
    add(1, "twin_prime_count", natural(0, 10 ** 5), "len(_r5_twin_primes({0}))",
        words("count|number of|how many", "twin prime"), TWIN_PRIMES)
    add(1, "goldbach_pair", natural(4, 10 ** 6), "_r5_goldbach_pair({0})",
        "goldbach|sum of two primes|two primes&sum|two prime numbers&sum|two primes&add|pair of primes|"
        + words("prime", "p + q|p+q|sum is|add up to|sum to"),
        GOLDBACH, valid=lambda v: v is not None)
    add(1, "classify_aliquot", natural(1, 10 ** 8),
        "(lambda s: 'perfect' if s == {0} else 'abundant' if s > {0} else 'deficient')(_r5_aliquot_sum({0}))",
        "abundant&deficient|perfect&deficient|perfect&abundant", ALIQUOT)
    add(1, "kaprekar_iterations", lambda n: is_int(n) and 1 <= n <= 9999 and len(set("%04d" % n)) > 1,
        "_r5_kaprekar_iterations({0})", "kaprekar|6174", KAPREKAR_ROUTINE)
    # Gray code: n XOR (n >> 1), and back by folding in every further right shift.
    add(1, "gray_encode", natural(0, 2 ** 64), "{0} ^ ({0} >> 1)", "gray")
    add(1, "gray_decode", natural(0, 2 ** 64),
        "functools.reduce(operator.xor, ({0} >> s for s in range(max({0}.bit_length(), 1))), 0)", "gray")
    add(1, "swap_nibbles", natural(0, 255), "(({0} & 0x0F) << 4) | ({0} >> 4)",
        "nibble|" + words("swap|exchange|interchange", "high 4 bits|high four bits|upper 4 bits|upper four bits"))
    # A fixed width comes from the description ("the 8 bits"); without one the reversal is of the binary
    # digits as written.
    add(2, "reverse_bits_width", lambda n, w: is_int(w) and 1 <= w <= 64 and is_int(n) and 0 <= n < 2 ** w,
        "int(format({0}, '0%db' % {1})[::-1], 2)", "revers&bit")
    add(1, "reverse_binary_digits", natural(1, 2 ** 64), "int(bin({0})[2:][::-1], 2)", "revers&bit|revers&binary")
    add(1, "next_same_bit_count", natural(1, 2 ** 62),
        "(lambda n, low: (n + low) | ((n ^ (n + low)) >> 2) // low)({0}, {0} & -{0})",
        "same number of 1|same number of one|same number of set bit|as many 1 bit|as many 1-bit|as many one bit|"
        "as many ones|as many set bit|same popcount|same hamming weight|same count of 1")
    add(2, "multiplicative_order",
        lambda a, n: is_int(a) and is_int(n) and a >= 1 and 2 <= n <= 10 ** 6 and math.gcd(a, n) == 1,
        "_r5_multiplicative_order({0}, {1})", "multiplicative order|order of&modulo|order&modulo|order of&mod ",
        MULTIPLICATIVE_ORDER)
    add(2, "round_significant",
        lambda x, k: is_num(x) and x != 0 and abs(x) < 10 ** 15 and is_int(k) and 1 <= k <= 15,
        "round({0}, {1} - 1 - math.floor(math.log10(abs({0}))))", "significant", kind="generic")
    add(2, "count_digit_upto", lambda n, d: is_int(n) and 1 <= n <= 10 ** 5 and is_int(d) and 0 <= d <= 9,
        "sum(str(k).count(str({1})) for k in range(1, {0} + 1))",
        words("digit", "1 to|from 1|all the integers|all integers|all numbers|written out|up to"))
    arrange = "rearrang|arrangement|permutation of its digits|formed from|formed by|can be formed"
    add(1, "smallest_arrangement", natural(1, 10 ** 18), "_r5_smallest_arrangement({0})",
        words("smallest|minimum|lowest|least", arrange), SMALLEST_ARRANGEMENT)
    add(1, "largest_arrangement", natural(1, 10 ** 18), "int(''.join(sorted(str({0}), reverse=True)))",
        words("largest|maximum|biggest|greatest|highest", arrange))
    base = lambda n, b: is_int(n) and 0 <= n <= 10 ** 18 and is_int(b) and 2 <= b <= 36  # noqa: E731
    add(2, "digits_in_base", base, "_r5_base_digits({0}, {1})", words("digit", "base"), BASE_DIGITS)
    add(2, "to_base", base, "''.join('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[d] for d in _r5_base_digits({0}, {1}))",
        words("convert|represent|written|write", "base"), BASE_DIGITS)

    add(1, "reverse_add_steps", natural(1, 10 ** 12), "_r5_reverse_add_steps({0})",
        "reverse-and-add|reverse and add|lychrel|" + words("revers", "add", "palindrom"), REVERSE_ADD)
    # Fractions as (numerator, denominator) pairs; Fraction keeps the sum in lowest terms with a positive denominator.
    fraction = lambda f: is_seq(f) and len(f) == 2 and is_int(f[0]) and is_int(f[1]) and f[1] != 0  # noqa: E731
    add(2, "add_fractions", lambda a, b: fraction(a) and fraction(b),
        "(lambda f: (f.numerator, f.denominator))(fractions.Fraction(*{0}) + fractions.Fraction(*{1}))",
        words("fraction", "add|sum|plus|total"))

    # ------------------------------------------------------------ named number properties (predicates)
    pred(1, "is_automorphic", natural(0, 10 ** 15), "str({0} * {0}).endswith(str({0}))",
         "automorphic|square ends with|square&ends with the|square&end with the")
    pred(1, "is_strong_number", natural(1, 10 ** 15), "sum(math.factorial(int(d)) for d in str({0})) == {0}",
         "strong number|factorion|krishnamurthy|peterson number|sum of the factorials|sum of factorials|"
         "factorials of its digits|factorial of each digit|factorial of its digits")
    pred(1, "is_kaprekar_number", natural(1, 10 ** 9), "_r5_is_kaprekar({0})", "kaprekar", KAPREKAR_NUMBER)
    pred(1, "is_spy_number", natural(1, 10 ** 18), "sum(map(int, str({0}))) == math.prod(map(int, str({0})))",
         "spy number|" + words("sum", "equal", "product of", "digit"))
    pred(1, "is_emirp", natural(1, 10 ** 7), "_r5_is_emirp({0})",
         "emirp|reversal is a different prime|reverse is a different prime|different prime", EMIRP)
    pred(1, "is_square_free", natural(1, 10 ** 10), "all({0} % (d * d) for d in range(2, math.isqrt({0}) + 1))",
         "square-free|squarefree|square free|not divisible by the square|not divisible by any square|"
         "not divisible by a square")
    pred(1, "has_alternating_bits", natural(1, 2 ** 64), "(lambda b: '00' not in b and '11' not in b)(bin({0})[2:])",
         "alternating bit|alternate bit|adjacent bits|bits&alternat|no two adjacent bits|consecutive bits")
    pred(1, "is_binary_palindrome", natural(0, 2 ** 64), "bin({0})[2:] == bin({0})[2:][::-1]",
         words("binary|bits", "palindrom|reads the same|same backward|same forward|same in reverse"))

    # ------------------------------------------------------------ formulas
    formula(1, "interior_angle", natural(3, 10 ** 6), "({0} - 2) * 180 / {0}", "interior angle")
    add(1, "interior_angle_sum", natural(3, 10 ** 6), "({0} - 2) * 180",
        "sum&interior angle|interior angles&sum|total&interior angle")
    add(1, "polygon_diagonals", natural(3, 10 ** 6), "{0} * ({0} - 3) // 2",
        "diagonal&polygon|diagonals of a|number of diagonals")
    formula(2, "bmi", lambda w, h: is_num(w) and is_num(h) and 0 < w <= 1000 and 0.2 <= h <= 3,
            "{0} / {1} ** 2", "body mass|bmi")
    temperature = lambda t: is_num(t) and -1000 <= t <= 10 ** 5  # noqa: E731
    formula(1, "fahrenheit_to_celsius", temperature, "({0} - 32) * 5 / 9", "fahrenheit&celsius|fahrenheit&centigrade")
    formula(1, "celsius_to_fahrenheit", temperature, "{0} * 9 / 5 + 32", "fahrenheit&celsius|fahrenheit&centigrade")
    # A change from an old to a new value, relative to the old one.
    formula(2, "percent_change", lambda old, new: is_num(old) and is_num(new) and old != 0,
            "({1} - {0}) / {0} * 100",
            words("percent|percentage", "change|increase|decrease|growth|rise|drop"))
    formula(2, "weighted_average", lambda v, w: nums(v) and nums(w) and len(v) == len(w) and sum(w) != 0,
            "sum(v * w for v, w in zip({0}, {1})) / sum({1})", "weighted")
    formula(2, "annulus_area", lambda big, small: is_num(big) and is_num(small) and big >= small >= 0,
            "math.pi * ({0} ** 2 - {1} ** 2)", "annulus|ring&circle|concentric|ring-shaped|between two circles")
    formula(2, "feet_inches_to_cm", lambda f, i: is_num(f) and is_num(i) and 0 <= f <= 100 and 0 <= i < 100,
            "({0} * 12 + {1}) * 2.54", "feet&inch|foot&inch|feet&centimet|feet&cm|inches&centimet")
    formula(1, "leibniz_pi", natural(1, 5000), "4 * sum((-1) ** k / (2 * k + 1) for k in range({0}))",
            "leibniz|madhava|gregory series")
    formula(1, "trimmed_mean", lambda x: nums(x, 3), "(sum({0}) - max({0}) - min({0})) / (len({0}) - 2)",
            "trimmed mean|trimmed average|olympic average|centered average|centred average|"
            + words("remov|exclud|without|ignor|except|discard|drop|trim|other than", "low|small|min",
                    "high|larg|max", "average|mean"))
    add(2, "clock_angle", lambda h, m: is_int(h) and is_int(m) and 0 <= h <= 24 and 0 <= m <= 59,
        "(lambda d: min(d, 360 - d))(abs({0} % 12 * 30 + {1} * 0.5 - {1} * 6))",
        "hour hand|minute hand|clock&hand|clock&angle")
    add(1, "lattice_points_in_circle", natural(0, 2000),
        "sum(2 * math.isqrt({0} * {0} - x * x) + 1 for x in range(-{0}, {0} + 1))",
        "lattice|integer coordinates&circle|integer points&circle|grid points&circle")

    def quadratic(a, b, c):
        return is_num(a) and is_num(b) and is_num(c) and a != 0 and max(abs(a), abs(b), abs(c)) < 10 ** 9
    # Real roots in increasing order and a repeated root once: the usual answer of "solve a*x**2 + b*x + c".
    add(3, "quadratic_roots", quadratic,
        "(lambda d: () if d < 0 else (-{1} / (2 * {0}),) if d == 0 else tuple(sorted(((-{1} - d ** 0.5) / (2 * {0}),"
        " (-{1} + d ** 0.5) / (2 * {0})))))({1} * {1} - 4 * {0} * {2})", "quadratic|roots of&equation|real roots")
    add(3, "loan_payment",
        lambda p, rate, n: is_num(p) and is_num(rate) and is_int(n) and p > 0 and 0 <= rate <= 100 and 1 <= n <= 1200,
        "(lambda r: {0} / {2} if r == 0 else {0} * r / (1 - (1 + r) ** -{2}))({1} / 100 / 12)",
        "loan&payment|monthly payment|mortgage|amortiz|installment|instalment")
    add(2, "stair_ways", lambda n, k: is_int(n) and is_int(k) and 0 <= n <= 1000 and 1 <= k <= 50,
        "_r5_stair_ways({0}, {1})", "stair|climb", STAIRS)
    add(2, "count_grid_rectangles", lambda r, c: is_int(r) and is_int(c) and 1 <= r <= 10 ** 6 and 1 <= c <= 10 ** 6,
        "({0} * ({0} + 1) // 2) * ({1} * ({1} + 1) // 2)", "rectangle&grid|rectangles in a")

    # ------------------------------------------------------------ calendar
    leap = "{1} % 4 == 0 and ({1} % 100 != 0 or {1} % 400 == 0)"
    add(2, "days_in_month", lambda m, y: is_int(m) and is_int(y) and 1 <= m <= 12 and 1 <= y <= 9999,
        "(31, 29 if " + leap + " else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[{0} - 1]",
        "days in&month|number of days&month|month&length|length&month")
    add(2, "leap_years_between", lambda a, b: is_int(a) and is_int(b) and 1 <= a <= b <= 10 ** 6,
        "(lambda f: f({1}) - f({0} - 1))(lambda y: y // 4 - y // 100 + y // 400)", "leap year")
    day_of_year = "day of the year|day of year|day number|ordinal day|ordinal date|within its year|within the year"
    add(3, "day_of_year", ymd, "datetime.date({0}, {1}, {2}).timetuple().tm_yday", day_of_year)
    add(1, "day_of_year_iso", iso_date, "datetime.date.fromisoformat({0}).timetuple().tm_yday", day_of_year)
    weekdays = "('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')"
    weekday_words = "day of the week|weekday|day of week|name of the day|which day"
    add(3, "weekday_name", ymd, weekdays + "[datetime.date({0}, {1}, {2}).weekday()]", weekday_words)
    add(1, "weekday_name_iso", iso_date, weekdays + "[datetime.date.fromisoformat({0}).weekday()]", weekday_words)
    pred(1, "is_weekend_iso", iso_date, "datetime.date.fromisoformat({0}).weekday() >= 5", "weekend|saturday|sunday")
    add(2, "days_between_iso", lambda a, b: iso_date(a) and iso_date(b),
        "abs((datetime.date.fromisoformat({1}) - datetime.date.fromisoformat({0})).days)",
        "days between|number of days between|days apart|" + words("days", "two dates|between dates|dates"))
    add(2, "add_days_iso", lambda d, n: iso_date(d) and is_int(n) and abs(n) <= 10 ** 6,
        "(datetime.date.fromisoformat({0}) + datetime.timedelta(days={1})).isoformat()",
        words("days", "after|later|before|earlier|add|plus|from"))
    # Completed years: one less while the birthday has not come yet in the given year.
    add(2, "age_on_date", lambda b, d: ymd_tuple(b) and ymd_tuple(d) and tuple(b) <= tuple(d),
        "{1}[0] - {0}[0] - (tuple({1}[1:]) < tuple({0}[1:]))",
        " age|years old|completed years|birthday|birth date|date of birth")

    # ------------------------------------------------------------ clock times and durations
    hms = "hh:mm:ss|h:mm:ss|hours, minutes and seconds|hours, minutes, and seconds|hours&minutes&seconds"
    duration = natural(0, 10 ** 9)
    add(1, "seconds_to_hms", duration, "'%d:%02d:%02d' % ({0} // 3600, {0} // 60 % 60, {0} % 60)", hms,
        kind="generic")
    add(1, "seconds_to_hhmmss", duration, "'%02d:%02d:%02d' % ({0} // 3600, {0} // 60 % 60, {0} % 60)", hms,
        kind="generic")
    add(1, "seconds_to_hms_tuple", duration, "({0} // 3600, {0} // 60 % 60, {0} % 60)", hms, kind="generic")
    add(1, "clock_to_seconds", lambda t: text(t) and re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", t) is not None,
        "(lambda h, m, s: h * 3600 + m * 60 + s)(*map(int, {0}.split(':')))",
        "hh:mm:ss|h:mm:ss|" + words("second", "convert|total|number of|how many|between|pass|elapse|difference"),
        kind="generic")
    add(1, "clock_to_minutes", lambda t: text(t) and re.fullmatch(r"\d{1,2}:\d{2}", t) is not None,
        "(lambda h, m: h * 60 + m)(*map(int, {0}.split(':')))",
        "hh:mm|h:mm|" + words("minute", "convert|total|number of|how many|between|pass|elapse|since midnight|"
                                         "after midnight|past midnight"), kind="generic")
    add(1, "to_12_hour", lambda t: text(t) and re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", t) is not None,
        "(lambda h, m: '%d:%02d %s' % (h % 12 or 12, m, 'AM' if h < 12 else 'PM'))(*map(int, {0}.split(':')))",
        "12-hour|12 hour|twelve-hour|twelve hour|am/pm|am or pm|a.m.|p.m.|'am'|'pm'")
    pred(1, "is_valid_time", text, "re.fullmatch(r'([01][0-9]|2[0-3]):[0-5][0-9]', {0}) is not None",
         words("valid|correct format|well-formed|well formed", "time|clock|hh:mm|24-hour|24 hour"))

    # ------------------------------------------------------------ number formats
    ordinal = "ordinal|'st'|'nd'|'rd'|1st&2nd"
    add(1, "ordinal", natural(0, 10 ** 15),
        "str({0}) + ('th' if {0} % 100 in (11, 12, 13) else {{1: 'st', 2: 'nd', 3: 'rd'}}.get({0} % 10, 'th'))",
        ordinal)
    add(1, "ordinal_suffix", natural(0, 10 ** 15),
        "'th' if {0} % 100 in (11, 12, 13) else {{1: 'st', 2: 'nd', 3: 'rd'}}.get({0} % 10, 'th')", ordinal)
    add(1, "number_to_words", natural(0, 10 ** 12 - 1), "_r5_number_words({0})",
        "in words|into words|to words|spell out|spelled out|spelt out|english words|number name|word form",
        NUMBER_WORDS)
    column = "column&label|column&title|column&letter|column name|column number|excel|spreadsheet"
    add(1, "column_label", natural(1, 10 ** 9), "_r5_column_label({0})", column, COLUMN_LABEL)
    add(1, "column_number", lambda s: text(s, 1, 7) and s.isascii() and s.isalpha(),
        "functools.reduce(lambda total, c: total * 26 + ord(c) - 64, {0}.upper(), 0)", column)
    add(1, "thousands_separator", lambda n: is_int(n) and abs(n) < 10 ** 30, "format({0}, ',')",
        "thousand|comma&group|commas&digit|separator&digit|digit group|group of three", kind="generic")

    # ------------------------------------------------------------ ciphers, codes and text conventions
    letters = lambda s: text(s, 1) and any(c.isalpha() for c in s)  # noqa: E731
    add(1, "atbash", letters,
        "{0}.translate(str.maketrans(string.ascii_letters, string.ascii_lowercase[::-1] + "
        "string.ascii_uppercase[::-1]))", "atbash|mirror&alphabet|reverse alphabet|reversed alphabet")
    key = lambda k: text(k, 1, 100) and k.isascii() and k.isalpha()  # noqa: E731
    add(2, "vigenere_encrypt", lambda t, k: letters(t) and key(k), "_r5_vigenere({0}, {1})", "vigen", VIGENERE)
    add(2, "vigenere_decrypt", lambda t, k: letters(t) and key(k), "_r5_vigenere({0}, {1}, -1)",
        "vigen&decrypt|vigen&decod|vigen&decipher", VIGENERE)
    add(2, "keyword_cipher", lambda t, k: letters(t) and key(k), "_r5_keyword_cipher({0}, {1})",
        "keyword cipher|key word cipher|keyed alphabet|cipher alphabet", KEYWORD_CIPHER)
    add(2, "rail_fence", lambda t, n: text(t, 2) and is_int(n) and 2 <= n <= 100, "_r5_rail_fence({0}, {1})",
        "rail fence|rail-fence|railfence|zigzag&cipher|zig-zag&cipher", RAIL_FENCE)
    add(1, "morse_encode", lambda s: text(s, 1, 200) and s.isascii() and s.replace(" ", "").isalnum(),
        "_r5_morse_encode({0})", "morse", MORSE_ENCODE)
    add(1, "morse_decode", lambda s: text(s, 1, 600) and set(s) <= set(".-/ ") and ("." in s or "-" in s),
        "_r5_morse_decode({0})", "morse", MORSE_DECODE)
    add(1, "pig_latin", lambda s: text(s, 1, 60) and s.isascii() and s.isalpha(), "_r5_pig_latin({0})",
        "pig latin|pig-latin|piglatin", PIG_LATIN)
    add(1, "pig_latin_sentence",
        lambda s: text(s, 3, 300) and s.isascii() and len(s.split()) > 1 and s.replace(" ", "").isalpha(),
        "' '.join(_r5_pig_latin(w) for w in {0}.split())", "pig latin|pig-latin|piglatin", PIG_LATIN)
    dna = lambda s: text(s, 1) and set(s.upper()) <= set("ACGT")  # noqa: E731
    add(1, "dna_complement", dna, "{0}.translate(str.maketrans('ACGTacgt', 'TGCAtgca'))",
        words("complement", "dna|strand|nucleotide|base"))
    add(1, "dna_reverse_complement", dna, "{0}[::-1].translate(str.maketrans('ACGTacgt', 'TGCAtgca'))",
        "reverse complement|reverse-complement|reversed complement")
    rle = "run-length|run length|rle|compress"
    add(1, "rle_char_count", lambda s: text(s, 1),
        "''.join(k + str(len(list(g))) for k, g in itertools.groupby({0}))",
        rle)
    add(1, "rle_char_count_skip_one", lambda s: text(s, 1),
        "''.join(k + (str(n) if n > 1 else '') for k, n in ((k, len(list(g))) for k, g in itertools.groupby({0})))",
        rle)
    # Count before character is also one step of the look-and-say sequence.
    add(1, "rle_count_char", lambda s: text(s, 1),
        "''.join(str(len(list(g))) + k for k, g in itertools.groupby({0}))",
        rle + "|look-and-say|look and say")
    decode = words("decod|decompress|expand|uncompress", "run-length|run length|rle|encoded|compressed")
    add(1, "rle_decode_char_count",
        lambda s: text(s, 2) and re.fullmatch(r"(\D\d*)+", s) is not None and any(c.isdigit() for c in s),
        "''.join(c * int(n or 1) for c, n in re.findall(r'(\\D)(\\d*)', {0}))", decode)
    add(1, "rle_decode_count_char", lambda s: text(s, 2) and re.fullmatch(r"(\d+\D)+", s) is not None,
        "''.join(c * int(n) for n, c in re.findall(r'(\\d+)(\\D)', {0}))", decode)
    add(1, "missing_letters", text, "''.join(c for c in string.ascii_lowercase if c not in {0}.lower())",
        words("letter", "missing|not occur|not appear|absent|not used|not present|do not occur|does not contain"))
    add(1, "slugify", lambda s: text(s, 1), "re.sub(r'[^a-z0-9]+', '-', {0}.lower()).strip('-')", "slug")
    add(1, "strip_html_tags", lambda s: text(s, 1) and "<" in s and ">" in s, "re.sub(r'<[^>]*>', '', {0})",
        "html|tags|markup", kind="generic")
    add(1, "sentence_case", letters,
        "re.sub(r'(^|[.!?]\\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), {0}.lower())",
        "sentence case|sentence-case|sentencecase|start of each sentence|beginning of each sentence|"
        "first letter of each sentence|first word of each sentence")
    add(1, "acronym", lambda s: letters(s) and len(re.split(r"[\s\-]+", s.strip())) > 1,
        "''.join(w[0].upper() for w in re.split(r'[\\s\\-]+', {0}) if w)", "acronym|initials")
    add(2, "word_wrap", lambda s, w: letters(s) and is_int(w) and 1 <= w <= 1000, "_r5_word_wrap({0}, {1})",
        words("wrap", "line|width|word"), WORD_WRAP, kind="generic")
    alternate = "alternat&case|alternating case|alternate&upper&lower|alternat&capital|alternately&upper"
    add(1, "alternate_case_letters", letters, "_r5_alternate_case({0}, True)", alternate, ALTERNATE_CASE)
    add(1, "alternate_case_positions", letters, "_r5_alternate_case({0}, False)", alternate, ALTERNATE_CASE)
    add(1, "file_extension", lambda s: text(s, 1, 300) and "." in s,
        "(lambda base: base.rsplit('.', 1)[1] if '.' in base else '')({0}.rsplit('/', 1)[-1].lstrip('.'))",
        "extension")
    add(1, "simplify_path", lambda s: text(s, 1) and s.startswith("/"), "_r5_simplify_path({0})",
        words("simplif|canonical|normaliz|normalis|resolve", "path"), SIMPLIFY_PATH)
    word_arg = lambda t, w: text(t, 1) and text(w, 1, 60) and re.fullmatch(r"\w+", w) is not None and w != t  # noqa
    count_word = "occur|times|how many|count|frequen"
    add(2, "count_whole_word", word_arg, "len(re.findall(r'\\b' + re.escape({1}) + r'\\b', {0}))",
        "whole word|as a word|separate word|standalone word|entire word|full word", kind="generic")
    add(2, "count_whole_word_nocase", word_arg,
        "len(re.findall(r'\\b' + re.escape({1}) + r'\\b', {0}, re.IGNORECASE))",
        words("word", count_word, "case|ignor|insensitive"), kind="generic")
    add(2, "count_distinct_substrings", lambda s, k: text(s, 1, 300) and is_int(k) and 1 <= k <= len(s),
        "len({{{0}[i:i + {1}] for i in range(len({0}) - {1} + 1)}})", words("distinct|unique|different", "substring"))
    add(1, "chars_once", lambda s: text(s, 2, 300), "''.join(c for c in {0} if {0}.count(c) == 1)",
        "exactly once|more than once|occur once|appear once|occurs once|appears once|only once|non-repeat|"
        "non repeat|not repeated")
    add(1, "min_add_to_balance", lambda s: text(s, 1) and set(s) <= set("()"), "_r5_min_add_to_balance({0})",
        words("parenthes|bracket", "minimum|fewest|least|smallest number", "add|insert|balance|valid"),
        MIN_ADD_PARENTHESES)
    add(2, "edit_distance", lambda a, b: text(a, 0, 60) and text(b, 0, 60) and len(a) + len(b) > 0,
        "_r5_edit_distance({0}, {1})",
        "levenshtein|edit distance|minimum number of edits|minimum edits|"
        + words("insertion|insert", "deletion|delet", "substitut|replace"), EDIT_DISTANCE)
    add(1, "longest_palindromic_substring", lambda s: text(s, 1, 300), "_r5_longest_palindrome({0})",
        words("longest|largest", "palindrom") + "|"
        + words("longest|largest", "substring", "reads the same|same backward|same forward|same in reverse"),
        LONGEST_PALINDROME)
    add(1, "longest_distinct_run", lambda s: text(s, 1) or flat(s), "_r5_longest_distinct_run({0})",
        words("longest", "without repeat|no repeat|non-repeating|without repetition|no character|distinct char|"
              "unique char|all distinct|no duplicate|more than once"), LONGEST_DISTINCT_RUN)
    add(2, "longest_common_substring", lambda a, b: text(a, 1, 60) and text(b, 1, 60),
        "_r5_longest_common_substring({0}, {1})",
        "longest common substring|longest common sub-string|longest common contiguous|longest common factor|"
        "longest&substring of both|longest&substring&both", LONGEST_COMMON_SUBSTRING)
    add(1, "smallest_rotation", lambda s: (text(s, 2, 200) or flat(s, 2, 200)),
        "min({0}[i:] + {0}[:i] for i in range(len({0})))",
        "lexicographic&rotat|smallest rotation|minimal rotation|minimum rotation|least rotation|"
        "smallest&rotation|alphabetically&rotat")
    add(1, "word_frequencies", letters,
        "dict(collections.Counter(w for w in (t.strip(string.punctuation) for t in {0}.lower().split()) if w))",
        words("word", "count|frequen|occur|times|tally|histogram", "lower|case|punctuat|ignor|insensitive"))
    add(1, "bigram_counts", lambda s: text(s, 1) and len(s.split()) > 1,
        "dict(collections.Counter(zip({0}.split(), {0}.split()[1:])))",
        "bigram|adjacent words|consecutive words|pairs of words|pair of words|word pairs|neighbouring words|"
        "neighboring words|adjacent word|consecutive word")
    add(1, "isbn13_check_digit", lambda s: text(s, 12, 12) and s.isdigit() and s.isascii(),
        "(10 - sum(int(d) * (3 if i % 2 else 1) for i, d in enumerate({0})) % 10) % 10", "isbn")

    add(1, "hex_to_rgb", lambda s: text(s, 6, 7) and re.fullmatch(r"#?[0-9a-fA-F]{6}", s) is not None,
        "tuple(int({0}.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))", words("hex", "rgb|red, green"))
    add(1, "keypad_digits", letters,
        "{0}.translate(str.maketrans(string.ascii_letters, '22233344455566677778889999' * 2))",
        "keypad|telephone|dial pad|t9|" + words("phone", "letter", "digit"))
    # The usual card mask keeps the last four digits; the description says so in words, not as a number.
    add(1, "mask_card_digits", lambda s: text(s, 5, 100) and sum(c.isdigit() for c in s) > 4,
        "_r5_mask_digits({0}, 4)", words("mask|hide|obscure|replace|star", "last four|last 4"), MASK_DIGITS)
    add(1, "decode_ways", lambda s: text(s, 1, 100) and s.isascii() and s.isdigit(), "_r5_decode_ways({0})",
        words("ways", "decod|letter code|letters|26"), DECODE_WAYS)

    # ------------------------------------------------------------ text predicates
    def same_kind(a, b):
        return (text(a, 1) and text(b, 1)) or (flat(a) and flat(b) and type(a) is type(b))

    pred(2, "is_subsequence", lambda a, b: same_kind(a, b) and len(a) <= len(b),
         "(lambda it: all(c in it for c in {0}))(iter({1}))",
         "subsequence|same order&not necessarily|in order&not necessarily|same relative order|appear in&same order")
    pred(1, "is_isogram", letters, "(lambda l: len(l) == len(set(l)))([c.lower() for c in {0} if c.isalpha()])",
         "an isogram|isogram&letter|isogram&case|isogram&word|isogram&phrase|isogram&string|isogram&ignor|"
         "isogram&space|isogram&hyphen|no repeating letter|without a repeating letter|no letter&more than once|"
         "no letter&repeat")
    pred(1, "can_form_palindrome", lambda s: text(s, 1) or flat(s),
         "sum(c % 2 for c in collections.Counter({0}).values()) <= 1",
         words("palindrom", "rearrang|permut|anagram|reorder|shuffl"))
    pred(1, "is_repeated_pattern", lambda s: text(s, 1), "len({0}) > 1 and {0} in ({0} + {0})[1:-1]",
         "repeated pattern|repeated substring|repeating substring|repetition of a substring|"
         + words("substring", "two or more times|several times|multiple times|more than once|copies") + "|"
         + words("repeat|repetition|copies of", "substring|sub-string|pattern|part of it|piece of it"))
    pred(1, "is_hex_color", text, "re.fullmatch(r'#([0-9a-fA-F]{{3}}|[0-9a-fA-F]{{6}})', {0}) is not None",
         words("hex", "colo") + "|colour code|color code|css colo")
    pred(1, "is_valid_mac", text, "re.fullmatch(r'[0-9A-Fa-f]{{2}}(:[0-9A-Fa-f]{{2}}){{5}}', {0}) is not None",
         "mac address|mac-address|hardware address")
    pred(1, "is_valid_identifier", text, "re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', {0}) is not None",
         "identifier|variable name")
    pred(1, "is_valid_semver", text,
         "re.fullmatch(r'(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)', {0}) is not None",
         "semver|semantic version|version number|version string|major.minor")
    pred(1, "passes_luhn", lambda s: (text(s, 1, 100) and set(s) <= set("0123456789 ")) or natural(0, 10 ** 30)(s),
         "_r5_luhn(str({0}))", "luhn|" + words("credit card|card number", "valid|check"), LUHN)
    pred(1, "is_valid_isbn10", lambda s: text(s, 10, 20), "_r5_isbn10_valid({0})", "isbn", ISBN10)
    pred(1, "is_tautogram", lambda s: letters(s) and len(s.split()) > 0,
         "len({{w[0].lower() for w in {0}.split()}}) <= 1",
         "tautogram|" + words("word", "begin with the same letter|start with the same letter|starts with the same "
                                      "letter|begins with the same letter|same first letter|same initial letter"))
    pred(1, "has_all_vowels", letters, "set('aeiou') <= set({0}.lower())",
         "all five vowels|all the vowels|every vowel|each vowel|each of the vowels|all of the vowels|"
         "contains all vowels|contain all vowels")
    pred(2, "one_edit_apart", lambda a, b: text(a, 0, 100) and text(b, 0, 100) and len(a) + len(b) > 0,
         "_r5_one_edit_apart({0}, {1})",
         "one edit|single edit|edit distance of one|edit distance of 1|exactly one edit|one edit away|"
         + words("one|single", "insert", "delet", "replac"), ONE_EDIT)
    pred(2, "wildcard_match", lambda s, p: text(s, 0, 100) and text(p, 1, 100) and ("*" in p or "?" in p),
         "_r5_wildcard_match({0}, {1})", "wildcard|wild card|glob|fnmatch", WILDCARD)
    pred(3, "is_interleaving", lambda a, b, c: text(a) and text(b) and text(c, 1, 200) and len(a) + len(b) == len(c),
         "_r5_is_interleaving({0}, {1}, {2})", "interleav", INTERLEAVING)

    # ------------------------------------------------------------ sequences: windows and rounding
    # "consecutive" alone is too common a word: a named window would outrank correct programs elsewhere.
    window = ("window|sliding|subarrays of size|subarrays of length|sublists of size|sublists of length|"
              "substrings of length|substrings of size|of size k|of length k|k consecutive")
    sliceable = lambda x, k: (seq(x, 1, 300) or text(x, 1, 300)) and is_int(k) and 1 <= k <= len(x)  # noqa: E731
    add(2, "windows", sliceable, "[{0}[i:i + {1}] for i in range(len({0}) - {1} + 1)]", window, kind="generic")
    add(2, "moving_average", lambda x, k: nums(x) and is_int(k) and 1 <= k <= len(x),
        "[sum({0}[i:i + {1}]) / {1} for i in range(len({0}) - {1} + 1)]",
        "moving average|rolling average|running average|sliding average|moving mean|rolling mean|"
        + words("window|consecutive", "average|mean"))
    add(2, "max_window_sum", lambda x, k: nums(x) and is_int(k) and 1 <= k <= len(x),
        "max(sum({0}[i:i + {1}]) for i in range(len({0}) - {1} + 1))",
        words("max|larg|great|high|best|biggest", "consecutive|window|contiguous|subarray of size|"
              "subarray of length|sublist of size|sublist of length", "sum"))
    add(1, "row_distinct_counts", lambda x: seq(x, 1, 300) and all(flat(r, 0) or text(r) for r in x),
        "type({0})(len(set(row)) for row in {0})",
        words("distinct|unique|different", "each|every|per|window|sublist|row|group"), kind="generic")
    add(2, "round_each", lambda x, k: nums(x) and any(isinstance(v, float) for v in x) and is_int(k) and 0 <= k <= 10,
        "type({0})(round(v, {1}) for v in {0})", ROUND, kind="generic")
    add(2, "round_values", lambda d, k: number_dict(d) and any(isinstance(v, float) for v in d.values())
        and is_int(k) and 0 <= k <= 10, "{{key: round(v, {1}) for key, v in {0}.items()}}", ROUND, kind="generic")

    # ------------------------------------------------------------ sequences: ranks, orders and positions
    add(1, "dense_rank", comparable,
        "(lambda r: type({0})(r[v] for v in {0}))({{v: i for i, v in enumerate(sorted(set({0})), 1)}})", "rank")
    add(1, "competition_rank", comparable,
        "(lambda s: type({0})(bisect.bisect_left(s, v) + 1 for v in {0}))(sorted({0}))", "rank")
    add(1, "argsort", lambda x: comparable(x, 2), "sorted(range(len({0})), key={0}.__getitem__)",
        "argsort|arg sort|indices that would|" + words("indices|indexes|index|positions", "sort"))
    target = lambda x, t: nums(x) and is_num(t)  # noqa: E731
    closest = "closest|nearest"
    add(2, "closest_index", target, "min(range(len({0})), key=lambda i: abs({0}[i] - {1}))",
        words("closest|nearest", "index|position"))
    add(2, "closest_value", target, "min({0}, key=lambda v: abs(v - {1}))",
        words(closest, "target|given value|given number|to a value|to the value|element"))
    add(1, "sort_by_abs", lambda x: nums(x, 2), "sorted({0}, key=abs)",
        words("absolute", "sort|order|arrang|rank"))
    add(1, "sort_by_abs_negative_first", lambda x: nums(x, 2), "sorted({0}, key=lambda v: (abs(v), v))",
        words("absolute", "sort|order|arrang", "negative"))
    add(1, "leaders", comparable, "type({0})(v for i, v in enumerate({0}) if all(v > w for w in {0}[i + 1:]))",
        "leader|than every element to|than all elements to|than all the elements to|than everything to|"
        "to their right|to its right")
    add(1, "next_greater", lambda x: nums(x), "_r5_next_greater({0}, False)",
        "next greater|next larger|next bigger|next higher|nearest greater|" + words("first greater|first larger",
                                                                                   "right|after"), NEXT_GREATER)
    add(1, "next_greater_distance", lambda x: nums(x), "_r5_next_greater({0}, True)",
        "warmer|days until|" + words("wait|until|how far|how many", "greater|larger|higher|warmer"), NEXT_GREATER)
    add(2, "find_sublist", lambda x, s: seq(x, 1) and seq(s, 1) and type(x) is type(s) and len(s) <= len(x),
        "next((i for i in range(len({0}) - len({1}) + 1) if {0}[i:i + len({1})] == {1}), -1)",
        words("sublist|subarray|sub-list|sub-array|contiguous", "index|position|occur|start|find|locat"))
    add(1, "duplicate_mask", flat, "[v in {0}[:i] for i, v in enumerate({0})]",
        "appears earlier|appeared earlier|appear earlier|seen before|occurred before|occurs earlier|"
        "occurred earlier|appeared before|appears before|earlier in the list|already appeared|already seen|"
        "already occurred|" + words("duplicate", "mask|flag|mark"))

    # ------------------------------------------------------------ sequences: selections and rearrangements
    add(1, "dedupe_keep_last", lambda x: flat(x, 2), "type({0})(list(dict.fromkeys({0}[::-1]))[::-1])",
        words("last occurr|last appear|last instance|last one|last copy|keep&last|keeping&last|retain&last",
              "duplicate|unique|distinct|repeat"))
    add(1, "evens_then_odds", lambda x: ints(x, 2),
        "type({0})([v for v in {0} if v % 2 == 0] + [v for v in {0} if v % 2])",
        words("even", "odd", "before|first|front|then|followed|segregat|separat|partition|rearrang|group|end"),
        kind="generic")
    add(1, "odds_then_evens", lambda x: ints(x, 2),
        "type({0})([v for v in {0} if v % 2] + [v for v in {0} if v % 2 == 0])",
        words("even", "odd", "before|first|front|then|followed|segregat|separat|partition|rearrang|group|end"),
        kind="generic")
    clamp = "clamp|clip|" + words("limit|restrict|bound|constrain|force", "range|between|interval")
    add(3, "clamp_each", lambda x, lo, hi: nums(x) and is_num(lo) and is_num(hi) and lo <= hi,
        "type({0})(min(max(v, {1}), {2}) for v in {0})", clamp, kind="generic")
    add(3, "clamp", lambda v, lo, hi: is_num(v) and is_num(lo) and is_num(hi) and lo <= hi,
        "min(max({0}, {1}), {2})", clamp, kind="generic")
    add(3, "swap_items", lambda x, i, j: seq(x, 2) and is_int(i) and is_int(j) and i != j
        and 0 <= i < len(x) and 0 <= j < len(x),
        "type({0})({0}[{2}] if k == {1} else {0}[{1}] if k == {2} else v for k, v in enumerate({0}))",
        "swap|exchange|interchange", kind="generic")
    add(2, "reverse_in_groups", lambda x, k: seq(x, 2) and is_int(k) and 2 <= k <= len(x),
        "type({0})(v for i in range(0, len({0}), {1}) for v in {0}[i:i + {1}][::-1])",
        words("revers", "group|chunk|every k|each k|block|batch|k element|k consecutive"), kind="generic")
    shift = words("shift", "fill|zero|without wrap|drop|pushed|discard|vacat|freed")
    add(2, "shift_right_fill", lambda x, k: nums(x) and is_int(k) and 1 <= k <= 1000,
        "(lambda k: type({0})([0] * k) + {0}[:len({0}) - k])(min({1}, len({0})))", shift, kind="generic")
    add(2, "shift_left_fill", lambda x, k: nums(x) and is_int(k) and 1 <= k <= 1000,
        "(lambda k: {0}[k:] + type({0})([0] * k))(min({1}, len({0})))", shift, kind="generic")
    add(2, "split_into_parts", lambda x, n: (seq(x) or text(x, 1)) and is_int(n) and 2 <= n <= 100,
        "(lambda q, r: [{0}[i * q + min(i, r):(i + 1) * q + min(i + 1, r)] for i in range({1})])"
        "(*divmod(len({0}), {1}))",
        "parts|evenly|as equal as possible|differ by at most one|equal-sized|equal size|roughly equal|nearly equal")
    add(2, "split_on_value", lambda x, v: isinstance(x, list) and 2 <= len(x) <= 500 and scalar(v) and v in x,
        "_r5_split_on({0}, {1})", "separator|delimiter|" + words("split", "at every|on every|wherever|occurrence|"
                                                                          "sublists|sub-lists"), SPLIT_ON,
        kind="generic")
    add(2, "interleave_rest", lambda x, y: seq(x, 0) and seq(y, 0) and type(x) is type(y) and len(x) + len(y) > 1,
        "[v for i in range(max(len({0}), len({1}))) for v in {0}[i:i + 1] + {1}[i:i + 1]]",
        "alternat|interleav|in turn|one by one from each|round robin|round-robin")
    add(2, "interleave_rest_text", lambda x, y: text(x) and text(y) and len(x) + len(y) > 1,
        "''.join({0}[i:i + 1] + {1}[i:i + 1] for i in range(max(len({0}), len({1}))))",
        "alternat|interleav|in turn|one by one from each|round robin|round-robin")
    # "Repeatedly" separates this from collapsing each run of duplicates to one copy (a round-3 rule).
    cancel = "pairs of equal|equal pairs|pair of equal|cancel|until no two|" + words(
        "adjacent|neighbo", "equal|identical", "remov|delet|eliminat") + "|" + words(
        "repeatedly|until no", "adjacent|neighbo|consecutive", "duplicate|equal|identical|pair")
    add(1, "cancel_adjacent_pairs", lambda x: seq(x, 2), "type({0})(_r5_cancel_pairs({0}))", cancel, CANCEL_PAIRS)
    add(1, "cancel_adjacent_pairs_text", lambda x: text(x, 2), "''.join(_r5_cancel_pairs({0}))", cancel,
        CANCEL_PAIRS)
    add(1, "sort_versions", lambda x: strs(x, 2) and all(re.fullmatch(r"\d+(\.\d+)*", v) for v in x),
        "sorted({0}, key=lambda v: tuple(int(p) for p in v.split('.')))", "version")
    add(1, "missing_between", lambda x: ints(x, 2) and max(x) - min(x) <= 10 ** 4,
        "(lambda s: [k for k in range(min({0}), max({0}) + 1) if k not in s])(set({0}))", "missing")
    add(1, "duplicate_and_missing",
        lambda x: ints(x, 2) and len(set(x)) == len(x) - 1 and all(1 <= v <= len(x) for v in x),
        "(lambda s: (next(v for i, v in enumerate({0}) if v in {0}[:i]), "
        "next(k for k in range(1, len({0}) + 1) if k not in s)))(set({0}))",
        words("missing", "duplicate|repeat|twice|appears two"))
    add(1, "modes", flat, "statistics.multimode({0})",
        "the mode|modes|mode of|mode value|most often|most frequent|most common|highest frequency|multimode")
    add(1, "modes_sorted", comparable,
        "(lambda c: sorted(v for v in c if c[v] == max(c.values())))(collections.Counter({0}))",
        words("the mode|modes|mode of|most often|most frequent|most common|highest frequency",
              "sort|increasing|ascending"))

    # ------------------------------------------------------------ sequences: named scans
    add(1, "prefix_products", nums, "type({0})(itertools.accumulate({0}, operator.mul))",
        "cumulative product|running product|prefix product|partial product|"
        + words("cumulative|running|prefix", "product"))
    add(1, "product_except_self", lambda x: nums(x, 1, 200),
        "type({0})(math.prod({0}[:i] + {0}[i + 1:]) for i in range(len({0})))",
        "except self|except itself|" + words("product", "other element|all other|except|excluding"))
    add(1, "max_gap", lambda x: nums(x, 2),
        "(lambda s: max((b - a for a, b in zip(s, s[1:])), default=0))(sorted({0}))",
        "maximum gap|max gap|largest gap|biggest gap|widest gap|"
        + words("largest|maximum|biggest|greatest", "difference|gap", "sorted|sorting"))
    add(1, "max_drawdown", nums, "max(peak - v for peak, v in zip(itertools.accumulate({0}, max), {0}))",
        "drawdown|draw-down|" + words("largest|biggest|maximum|greatest", "drop|fall|decline"))
    add(1, "trapped_water", lambda x: nums(x) and all(v >= 0 for v in x), "_r5_trapped_water({0})",
        "trap&water|rain water|rainwater|water trapped|trapped water", TRAPPED_WATER)
    add(1, "h_index", lambda x: ints(x) and all(v >= 0 for v in x),
        "sum(c >= i for i, c in enumerate(sorted({0}, reverse=True), 1))", "h-index|h index|hirsch")
    add(1, "longest_increasing_run", nums, "_r5_longest_increasing_run({0})",
        words("longest|maximum length|max length", "increasing|ascending",
              "run|contiguous|consecutive|subarray|sublist|streak|stretch|substring") + "|"
        + words("longest", "run|streak|stretch|contiguous|consecutive", "greater than the one before|"
                "greater than the previous|larger than the previous|bigger than the previous"), INCREASING_RUN)
    run = words("longest|maximum|max|biggest|largest", "run|streak|stretch|consecutive|in a row",
                "equal|same|identical|repeated|repeating")
    add(1, "longest_equal_run", lambda x: flat(x) or text(x, 1),
        "max(((k, len(list(g))) for k, g in itertools.groupby({0})), key=lambda p: p[1])", run)
    add(1, "longest_equal_run_length", lambda x: flat(x) or text(x, 1),
        "max(len(list(g)) for _, g in itertools.groupby({0}))", run)
    add(2, "count_subarrays_with_sum", lambda x, k: nums(x, 1, 1000) and is_num(k),
        "_r5_count_subarray_sums({0}, {1})", words("count|number of|how many", "subarray|sublist|contiguous|"
                                                                               "sub-array|sub-list",
                                                   "sum|add up|total"), SUBARRAY_SUM_COUNT)
    add(2, "count_pairs_sum_divisible", lambda x, k: ints(x, 2, 150) and is_int(k) and k >= 1,
        "sum((a + b) % {1} == 0 for a, b in itertools.combinations({0}, 2))", words("pair", "divisible"))
    add(1, "min_partition_difference",
        lambda x: ints(x, 1, 40) and all(v >= 0 for v in x) and sum(x) <= 20000, "_r5_min_partition_difference({0})",
        "partition problem|minimum subset sum difference|"
        + words("two groups|two subsets|two sets|two parts|two piles",
                                                                   "difference"), PARTITION_DIFFERENCE)
    add(2, "kth_permutation", lambda n, k: is_int(n) and is_int(k) and 1 <= n <= 12 and 1 <= k <= math.factorial(n),
        "_r5_kth_permutation({0}, {1})",
        "k-th permutation|kth permutation|nth permutation|n-th permutation|permutation sequence|"
        + words("permutation", "lexicographic"), KTH_PERMUTATION)
    add(2, "josephus_order", lambda n, k: is_int(n) and is_int(k) and 1 <= n <= 2000 and 1 <= k <= 10 ** 6,
        "_r5_josephus_order({0}, {1})", "josephus|counting-out|" + words("circle", "eliminat|removed|every k"),
        JOSEPHUS)
    add(2, "josephus_survivor", lambda n, k: is_int(n) and is_int(k) and 1 <= n <= 10 ** 5 and 1 <= k <= 10 ** 6,
        "functools.reduce(lambda s, m: (s + {1}) % m, range(2, {0} + 1), 0) + 1",
        "josephus|counting-out|" + words("circle", "eliminat|removed|every k"))
    rpn_token = lambda t: re.fullmatch(r"-?\d+|[-+*/]", t) is not None  # noqa: E731
    add(1, "eval_rpn", lambda x: strs(x) and all(rpn_token(t) for t in x), "_r5_eval_rpn({0})",
        "reverse polish|rpn|postfix", RPN)
    add(1, "eval_rpn_text", lambda s: text(s, 1) and all(rpn_token(t) for t in s.split()),
        "_r5_eval_rpn({0}.split())",
        "reverse polish|rpn|postfix", RPN)

    # ------------------------------------------------------------ sequence predicates
    pred(1, "is_mountain", lambda x: nums(x, 3), "_r5_is_mountain({0})",
         "mountain|single peak|one peak|" + words("strictly increas", "strictly decreas"), MOUNTAIN)
    pred(1, "has_increasing_triplet", lambda x: comparable(x, 3), "_r5_has_increasing_triplet({0})",
         "increasing triplet|increasing subsequence of length 3|i < j < k|" + words("three|triplet", "increasing"),
         INCREASING_TRIPLET)
    pred(2, "has_nearby_duplicate", lambda x, k: flat(x, 2) and is_int(k) and k >= 1,
         "any({0}[i] in {0}[max(0, i - {1}):i] for i in range(len({0})))",
         "nearby duplicate|" + words("duplicate|equal", "at most|within|distance|apart|differ by"))
    pred(2, "has_pair_with_difference", lambda x, k: nums(x, 2, 200) and is_num(k),
         "any(a - b == {1} for i, a in enumerate({0}) for j, b in enumerate({0}) if i != j)",
         words("differ|difference", "pair|two element|two number|two values|two different|two entries"))
    pred(1, "is_arithmetic", lambda x: nums(x, 2), "len({{b - a for a, b in zip({0}, {0}[1:])}}) <= 1",
         "arithmetic progression|arithmetic sequence|common difference|same difference")
    pred(1, "can_form_arithmetic", lambda x: nums(x, 2),
         "(lambda s: len({{b - a for a, b in zip(s, s[1:])}}) <= 1)(sorted({0}))",
         words("arithmetic", "rearrang|reorder|any order|permut|sorted|reordered"))
    pred(1, "is_permutation_of_range", ints, "sorted({0}) == list(range(1, len({0}) + 1))",
         words("each|every|all|exactly once", "1 to n|from 1 to|1 and n|1..n") + "|permutation of 1|"
         # Longer than "is a permutation of" (two arguments compared), which names a round-3 rule.
         + words("permutation of|permutation", "1 to n|1..n|1 and n|1 through n|1 up to n"))
    pred(1, "is_zigzag", lambda x: nums(x, 2),
         "(lambda d: all(d) and all((a > 0) != (b > 0) for a, b in zip(d, d[1:])))([b - a for a, b in zip({0}, "
         "{0}[1:])])", "zigzag|zig-zag|zig zag|wiggle|" + words("alternat", "sign|up and down|up, down"))
    overlap = lambda x: pairs(x) and all(p[0] <= p[1] for p in x)  # noqa: E731
    pred(1, "has_overlap", overlap, "(lambda s: any(b[0] < a[1] for a, b in zip(s, s[1:])))(sorted({0}))",
         "overlap")
    pred(1, "has_overlap_closed", overlap, "(lambda s: any(b[0] <= a[1] for a, b in zip(s, s[1:])))(sorted({0}))",
         words("overlap", "touch|share an endpoint|share a point|inclusive|closed"))
    pred(1, "is_prefix_free", lambda x: strs(x, 1, 200),
         "not any(i != j and b.startswith(a) for i, a in enumerate({0}) for j, b in enumerate({0}))",
         "prefix-free|prefix free|prefix code|prefix of another|prefix of any other")

    # ------------------------------------------------------------ matrices and grids
    cells = lambda x: grid(x, scalar, 50)  # noqa: E731
    add(1, "spiral_order", cells, "_r5_spiral_order({0})", "spiral", SPIRAL)
    add(2, "matrix_multiply", lambda a, b: grid(a, is_num, 30) and grid(b, is_num, 30) and len(a[0]) == len(b),
        "[[sum(a * b for a, b in zip(row, col)) for col in zip(*{1})] for row in {0}]",
        "matmul|multiply&matrices|multiply two matrices|product of two matrices|" + words("matri", "multipl|product"))
    add(1, "determinant", lambda x: square(x, is_num, 10), "_r5_determinant({0})", "determinant", DETERMINANT)
    add(1, "rotate_clockwise", cells, "[list(row) for row in zip(*{0}[::-1])]",
        words("rotat", "clockwise|90|matri|grid|image|square"))
    add(1, "rotate_counterclockwise", cells, "[list(row) for row in zip(*{0})][::-1]",
        "counterclockwise|counter-clockwise|anticlockwise|anti-clockwise|counter clockwise|anti clockwise")
    add(1, "life_step", lambda x: grid(x, bit, 40), "_r5_life_step({0})",
        "game of life|conway|next generation|life&generation", LIFE)
    add(1, "count_islands", bit_grid, "_r5_islands({0})", "island", ISLANDS)
    add(1, "island_perimeter", bit_grid, "_r5_island_perimeter({0})",
        words("perimeter", "island|land") + "|coastline", ISLAND_PERIMETER)
    add(1, "matrix_to_sparse", lambda x: grid(x, is_num, 50),
        "{{(r, c): v for r, row in enumerate({0}) for c, v in enumerate(row) if v != 0}}",
        "sparse|non-zero entries|nonzero entries|non-zero elements|nonzero elements|non-zero values")
    add(1, "saddle_points", lambda x: grid(x, is_num, 30),
        "[(r, c) for r, row in enumerate({0}) for c, v in enumerate(row) if v == max(row) and "
        "v == min(other[c] for other in {0})]", "saddle")
    add(1, "anti_diagonals", cells,
        "[[{0}[r][k - r] for r in range(len({0})) if 0 <= k - r < len({0}[0])] for k in range(len({0}) + "
        "len({0}[0]) - 1)]", "anti-diagonal|antidiagonal|anti diagonal|diagonal order|" + words(
            "diagonal", "group|traversal|each diagonal|every diagonal"))
    add(1, "minesweeper",
        lambda x: strs(x, 1, 40) and all(set(r) <= set("*. ") for r in x) and any("*" in r for r in x),
        "_r5_minesweeper({0})", "minesweeper|mine sweeper| mines|mine&count|mine&neighbo", MINESWEEPER)
    maze = lambda x: (strs(x, 1, 40) and len({len(r) for r in x}) == 1 and all(set(r) <= set(".#") for r in x)
                      and len(x[0]) > 0) or grid(x, bit, 40)  # noqa: E731
    add(1, "grid_shortest_path", maze, "_r5_grid_shortest_path({0})",
        words("shortest|fewest moves|fewest steps|minimum number of moves|smallest number of moves|minimum moves|"
              "least moves|minimum steps", "grid|maze|matrix|board|cell"), GRID_PATH)
    add(1, "paths_avoiding_obstacles", lambda x: grid(x, bit, 50), "_r5_paths_avoiding({0})",
        "obstacle|" + words("paths", "avoid|blocked|never enter|right or down|right and down"), OBSTACLE_PATHS)
    add(1, "tictactoe_winner", lambda b: seq(b, 3, 3) and all((text(r, 3, 3) or seq(r, 3, 3)) for r in b),
        "_r5_tictactoe_winner({0})", "tic-tac-toe|tic tac toe|tictactoe|noughts and crosses", TICTACTOE)
    add(1, "parse_matrix_text", lambda s: text(s, 1) and re.fullmatch(r"[ \t]*-?\d+([ \t]+-?\d+)*"
                                                                      r"(\n[ \t]*-?\d+([ \t]+-?\d+)*)*[ \t]*", s),
        "[[int(v) for v in line.split()] for line in {0}.strip().split('\\n')]",
        words("matri|grid|rows|lines", "parse|convert|string|text|read"), kind="generic")
    add(2, "dot_product", lambda a, b: nums(a) and nums(b) and len(a) == len(b),
        "sum(a * b for a, b in zip({0}, {1}))", "dot product|inner product|scalar product|dot-product")
    add(2, "sparse_dot_product", lambda a, b: number_dict(a) and number_dict(b),
        "sum(v * {1}[k] for k, v in {0}.items() if k in {1})", "dot product|inner product|scalar product|dot-product")
    pred(1, "is_sudoku", lambda x: square(x, is_int, 16) and len(x) in (4, 9, 16), "_r5_is_sudoku({0})", "sudoku",
         SUDOKU)
    pred(1, "is_toeplitz", lambda x: grid(x, scalar, 50),
         "all({0}[r][c] == {0}[r - 1][c - 1] for r in range(1, len({0})) for c in range(1, len({0}[0])))",
         "toeplitz|" + words("diagonal", "same|equal|constant", "matri"))
    pred(1, "is_symmetric_matrix", lambda x: square(x, scalar, 30),
         "all({0}[i][j] == {0}[j][i] for i in range(len({0})) for j in range(len({0})))",
         words("symmetric", "matri|transpose") + "|own transpose|its transpose|equal to the transpose")

    # ------------------------------------------------------------ intervals, knapsack and sums of records
    add(1, "merge_intervals", lambda x: pairs(x) and all(p[0] <= p[1] for p in x), "_r5_merge_intervals({0})",
        words("interval|range|overlap", "merge|combin|union|join|consolidat|collapse"), MERGE_INTERVALS)
    add(2, "knapsack", lambda x, c: pairs(x, natural(0, 10 ** 6), 1, 100) and is_int(c) and 0 <= c <= 10 ** 4,
        "_r5_knapsack({0}, {1})", "knapsack", KNAPSACK)
    add(3, "knapsack_lists", lambda w, v, c: ints(w, 1, 100) and ints(v, 1, 100) and len(w) == len(v)
        and min(w) >= 0 and is_int(c) and 0 <= c <= 10 ** 4,
        "_r5_knapsack(list(zip({0}, {1})), {2})", "knapsack", KNAPSACK)
    add(2, "greedy_change", lambda a, d: is_int(a) and 0 <= a <= 10 ** 9 and ints(d, 1, 30) and min(d) > 0,
        "_r5_greedy_change({0}, {1})",
        words("greed|largest coin|largest denomination|break", "coin|change|denomination")
        + "|denomination&count|coin&breakdown", GREEDY_CHANGE)
    add(2, "bucket_histogram", lambda x, w: ints(x) and min(x) >= 0 and is_int(w) and w >= 1,
        "dict(collections.Counter(v // {1} * {1} for v in {0}))", "histogram|bucket|bins of|bin width|binning")
    add(1, "shoelace_area", lambda x: pairs(x, is_num, 3),
        "abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip({0}, {0}[1:] + {0}[:1]))) / 2",
        "shoelace|" + words("area", "vertices|vertex|coordinates|points|polygon given"))

    # ------------------------------------------------------------ dictionaries
    small_dict = lambda d: isinstance(d, dict) and 1 <= len(d) <= 200 and all(scalar(v) for v in d.values())  # noqa
    invert = words("invert|inverse|revers|swap|flip", "dict|key|value") + "|" + words(
        "keys", "same value|that value|by value|share|sharing")
    add(1, "invert_dict_grouped", small_dict,
        "{{v: [k for k in {0} if {0}[k] == v] for v in dict.fromkeys({0}.values())}}", invert)
    add(1, "invert_dict_grouped_sorted", small_dict,
        "{{v: sorted(k for k in {0} if {0}[k] == v) for v in dict.fromkeys({0}.values())}}",
        words("invert|inverse|revers|swap|flip|keys", "sort"))
    first = "first letter|first character|initial letter|starting letter|first char|initial"
    add(1, "group_by_first_letter", lambda x: strs(x, 2) and all(x), "_r5_group_by_first({0}, False)",
        words(first, "group|dictionar|map|bucket"), GROUP_BY_FIRST)
    add(1, "group_by_first_letter_lower", lambda x: strs(x, 2) and all(x), "_r5_group_by_first({0}, True)",
        words(first, "group|dictionar|map|bucket", "lower|case"), GROUP_BY_FIRST)
    add(1, "group_anagrams", lambda x: strs(x, 2), "_r5_group_anagrams({0})",
        words("anagram", "group|together|cluster"),
        GROUP_ANAGRAMS)
    def nested_dict(d):
        return isinstance(d, dict) and 1 <= len(d) <= 200 and any(isinstance(v, dict) for v in d.values())
    flat_dict = words("flatten|flat|single level|single-level|one level", "dict|nested")
    add(1, "flatten_dict", nested_dict, "_r5_flatten_dict({0})", flat_dict, FLATTEN_DICT)
    add(2, "flatten_dict_with", lambda d, s: nested_dict(d) and text(s, 1, 3), "_r5_flatten_dict({0}, {1})",
        flat_dict,
        FLATTEN_DICT)
    add(2, "get_path", lambda d, keys: nested_dict(d) and flat(keys, 1, 20), "_r5_get_path({0}, {1})",
        "key path|path of keys|chain of keys|sequence of keys|list of keys|" + words("nested", "key", "follow|path|"
                                                                                       "sequence|list"), GET_PATH)
    add(2, "deep_merge",
        lambda a, b: isinstance(a, dict) and isinstance(b, dict) and (nested_dict(a) or nested_dict(b)),
        "_r5_deep_merge({0}, {1})", "deep merge|deep-merge|" + words("merge", "recursiv|nested"), DEEP_MERGE)
    add(1, "deep_reverse", lambda x: seq(x) and any(is_seq(v) for v in x), "_r5_deep_reverse({0})",
        words("revers", "every level|all level|each level|nested|recursiv|inner|deep"), DEEP_REVERSE)
    add(1, "swap_dict_levels", lambda d: isinstance(d, dict) and 1 <= len(d) <= 200
        and all(isinstance(v, dict) for v in d.values()) and any(d.values()), "_r5_swap_levels({0})",
        words("swap|transpose|invert|exchange|pivot", "level|nested|inner|outer"), SWAP_LEVELS)
    add(1, "columns_to_rows", lambda d: isinstance(d, dict) and 1 <= len(d) <= 100
        and all(seq(v, 1) for v in d.values()) and len({len(v) for v in d.values()}) == 1,
        "[dict(zip({0}, values)) for values in zip(*{0}.values())]",
        words("column", "row|record", "dict") + "|dictionary of lists&list of dictionaries")
    add(1, "rows_to_columns", lambda x: records(x) and all(r.keys() == x[0].keys() for r in x),
        "{{k: [row[k] for row in {0}] for k in {0}[0]}}",
        words("column", "row|record", "dict") + "|list of dictionaries&dictionary of lists")
    add(2, "count_by_key", lambda x, k: records(x) and scalar(k) and all(k in r and scalar(r[k]) for r in x),
        "dict(collections.Counter(record[{1}] for record in {0}))",
        words("count|number of|how many|frequen|tally|occurr", "field|key|attribute|property|column"))
    lists_dict = lambda d: isinstance(d, dict) and 1 <= len(d) <= 200 and all(nums(v) for v in d.values())  # noqa
    # One aggregate per key; "mean" is matched as a word ("meaning" must not name it).
    per_key = "each|every|per|for all"
    add(1, "dict_value_means", lists_dict, "{{k: sum(v) / len(v) for k, v in {0}.items()}}",
        words("average|mean of|the mean|mean value|means of|arithmetic mean|mean for", per_key))
    add(1, "dict_value_sums", lists_dict, "{{k: sum(v) for k, v in {0}.items()}}", words("sum|total", per_key))
    add(2, "rename_keys", lambda d, m: isinstance(d, dict) and isinstance(m, dict) and 1 <= len(m) <= 200
        and any(k in d for k in m), "{{{1}.get(k, k): v for k, v in {0}.items()}}", "renam")
    add(1, "positions_of_each", lambda x: text(x, 1, 300) or flat(x, 1, 300),
        "{{k: [i for i, v in enumerate({0}) if v == k] for k in dict.fromkeys({0})}}",
        words("position|indices|index|indexes|locations", "each|every|all"))
    add(1, "parse_query_string", lambda s: text(s, 1) and "=" in s, "_r5_parse_query({0})",
        "query string|query&url|url&parameter|key=value|query parameters", QUERY_STRING)
    add(1, "parse_csv_records", lambda s: text(s, 3) and "\n" in s,
        "(lambda lines: [dict(zip(lines[0].split(','), line.split(','))) for line in lines[1:]])({0}.split('\\n'))",
        "csv|comma-separated|comma separated")
    pred(2, "is_sub_dict", lambda a, b: isinstance(a, dict) and isinstance(b, dict) and len(a) >= 1,
         "all(k in {1} and {1}[k] == v for k, v in {0}.items())",
         "sub-dict|subdict|sub dict|sub-dictionary|subdictionary|" + words("subset", "dictionar") + "|"
         + words("key-value pair", "also|contained|appear"))
    pred(1, "is_injective", small_dict, "len(set({0}.values())) == len({0})",
         "injective|one-to-one|one to one|" + words("no two keys|values", "same value|unique|distinct|different"))

    # ------------------------------------------------------------ trees and graphs
    add(1, "inorder", real_tree, "_r5_tree_order({0}, 'in')", "inorder|in-order|in order traversal", TREE_ORDER)
    add(1, "preorder", real_tree, "_r5_tree_order({0}, 'pre')", "preorder|pre-order", TREE_ORDER)
    add(1, "postorder", real_tree, "_r5_tree_order({0}, 'post')", "postorder|post-order", TREE_ORDER)
    pred(1, "is_bst", lambda x: real_tree(x), "_r5_is_bst({0})", "binary search tree|a bst|is bst| bst ", IS_BST)
    starts = lambda g, s: adjacency(g) and scalar(s) and s in g  # noqa: E731
    add(2, "bfs_order", starts, "_r5_bfs_order({0}, {1})", "breadth-first|breadth first|bfs", BFS)
    add(2, "dfs_order", lambda g, s: starts(g, s) and len(g) <= 500, "_r5_dfs_order({0}, {1})",
        "depth-first|depth first|dfs", DFS)
    names = lambda g: adjacency(g) and comparable(list(g)) and all(flat(v, 0) for v in g.values())  # noqa: E731
    topological = "topolog|dependenc|prerequisite|depends on|depend on"
    add(1, "topological_order", names, "_r5_topological_order({0})", topological, TOPOLOGICAL)
    add(1, "topological_order_edges", names, "_r5_topological_order_edges({0})", topological, TOPOLOGICAL_EDGES)

    def weighted(g):
        return (isinstance(g, dict) and 1 <= len(g) <= 300 and all(
            (isinstance(e, dict) and all(is_num(w) and w >= 0 for w in e.values()))
            or (is_seq(e) and all(is_seq(p) and len(p) == 2 and is_num(p[1]) and p[1] >= 0 for p in e))
            for e in g.values()))
    add(2, "shortest_distances", lambda g, s: weighted(g) and scalar(s) and s in g,
        "_r5_shortest_distances({0}, {1})",
        "dijkstra|" + words("shortest|minimum|smallest", "distance|path|cost", "weight|graph|vertex|vertices|node"),
        DIJKSTRA)
    add(3, "all_paths", lambda g, a, b: adjacency(g, 12) and scalar(a) and scalar(b) and a in g,
        "_r5_all_paths({0}, {1}, {2})", "all paths|every path|all possible paths|all the paths|each path", ALL_PATHS)
    add(1, "undirected_adjacency", lambda x: pairs(x, scalar) and comparable([v for p in x for v in p]),
        "_r5_undirected_adjacency({0})", "undirected|bidirectional|both directions", UNDIRECTED)
    add(2, "path_from_root",
        lambda d, n: isinstance(d, dict) and 1 <= len(d) <= 1000 and scalar(n) and n in d and None in d.values(),
        "_r5_path_from_root({0}, {1})", words("path|chain", "root") + "|ancestor|lineage", ROOT_PATH)
    pred(1, "is_bipartite", lambda g: adjacency(g) or (seq(g, 1, 300) and all(ints(v, 0) for v in g)),
         "_r5_is_bipartite({0})", "bipartite|two colo|2-colo|two-colo", BIPARTITE)
    pred(1, "has_directed_cycle", lambda g: adjacency(g) and all(flat(v, 0) for v in g.values()),
         "_r5_has_directed_cycle({0})", words("cycle|cyclic", "graph|directed|dependenc|edge|vertex|vertices|node"),
         DIRECTED_CYCLE)

    # ------------------------------------------------------------ more words for existing operators
    # Each keeps its behaviour; the descriptions of the bank name these rules with words they lacked.
    for existing, extra in [
            # The specificity tie-break let "two&equal length" (same_length) beat "isomorphic".
            ("is_isomorphic/2", "isomorphic&replace|isomorphic&consistent|isomorphic&map|isomorphic&equal length"),
            ("strictly_increasing/1", "strictly increase|strictly ascend|strictly rise|strictly grow"),
            ("strictly_decreasing/1", "strictly decrease|strictly descend|strictly fall"),
            ("is_fibonacci_number/1", "belongs to the fibonacci|in the fibonacci sequence|part of the fibonacci|"
                                      "member of the fibonacci|appears in the fibonacci|fibonacci number"),
            ("num_min_squares/1", "number of perfect squares|perfect squares&add up|perfect squares&sum to|"
                                  "squares&add up to|perfect squares&add to"),
            # Adjacent swaps that sort a list are exactly its inversions.
            ("seq_count_inversions/1", "inversion|swaps of neighbo|neighbouring swaps|neighboring swaps|"
                                       "adjacent swaps|swaps of adjacent|bubble sort&swap"),
            # "the first occurrence of the smallest element" is its index.
            ("argmin/1", "occurrence&smallest|occurrence&minimum"),
            ("argmax/1", "occurrence&largest|occurrence&maximum"),
            ("is_palindrome/1", "reads the same backward|reads the same forward|same forwards and backwards|"
                                "same when reversed")]:
        try:
            lib.also_named(existing, extra)
        except KeyError:        # the pack can be loaded without the round-3 packs it extends
            pass
