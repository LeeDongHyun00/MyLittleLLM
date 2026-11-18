"""Round 3 operators for strings: regex extraction and substitution, character classes, word edits, conversions."""
import collections
import itertools
import re
import string
import warnings

# Every name starts with txt_ (helpers with _txt_): packs are written in parallel and a name clash stops the loader.
# Multi-line rules are kept as source text because the same text is pasted into generated answers.
ROMAN = '''def _txt_roman_to_int(s):
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i, c in enumerate(s):
        # A smaller numeral in front of a larger one is subtracted (IV is 4).
        if i > 0 and values[c] > values[s[i - 1]]:
            total += values[c] - 2 * values[s[i - 1]]
        else:
            total += values[c]
    return total
'''
REVERSE_VOWELS = '''def _txt_reverse_vowels(s):
    vowels = [c for c in s if c in 'aeiouAEIOU']
    return ''.join(vowels.pop() if c in 'aeiouAEIOU' else c for c in s)
'''
LCS = '''def _txt_lcs(x, y):
    # One row of the usual table is enough: `diagonal` carries the value of the previous row.
    row = [0] * (len(y) + 1)
    for a in x:
        diagonal = 0
        for j, b in enumerate(y, 1):
            diagonal, row[j] = row[j], diagonal + 1 if a == b else max(row[j], row[j - 1])
    return row[-1]
'''
LCS3 = '''def _txt_lcs3(x, y, z):
    table = [[[0] * (len(z) + 1) for _ in range(len(y) + 1)] for _ in range(len(x) + 1)]
    for i, a in enumerate(x, 1):
        for j, b in enumerate(y, 1):
            for k, c in enumerate(z, 1):
                if a == b == c:
                    table[i][j][k] = table[i - 1][j - 1][k - 1] + 1
                else:
                    table[i][j][k] = max(table[i - 1][j][k], table[i][j - 1][k], table[i][j][k - 1])
    return table[-1][-1][-1]
'''
REPEATING = '''def _txt_repeating_subsequence(s):
    # Longest common subsequence of the string with itself, never pairing a position with itself.
    n = len(s)
    table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == s[j - 1] and i != j:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i][j - 1], table[i - 1][j])
    return table[n][n]
'''
BRACKET_SWAPS = '''def _txt_bracket_swaps(s):
    left = right = swaps = imbalance = 0
    for c in s:
        if c == '[':
            left += 1
            if imbalance > 0:
                # Every unmatched closing bracket seen so far has to travel past this opening one.
                swaps += imbalance
                imbalance -= 1
        elif c == ']':
            right += 1
            imbalance = right - left
    return swaps
'''


def register(lib):
    need, is_int = lib.need, lib.is_int
    scope = {"re": re, "itertools": itertools, "collections": collections, "string": string}

    def add(arity, name, guard, template, keywords, *, gated=True, knowledge=False, helper=None, cost=1.0,
            quiet=False):
        """Register an operator whose search function is its own template, compiled once.

        Compiling the emitted expression keeps the two forms of an operator identical by construction;
        the guard only decides which inputs the search tries it on.
        """
        if helper:
            function, helper = lib.helper(helper)
            scope[function.__name__] = function
        params = ["_x0", "_x1", "_x2"][:arity]
        run = eval(f"lambda {', '.join(params)}: {template.format(*params)}", scope)

        def fn(*args):
            need(guard(*args))
            if not quiet:
                return run(*args)
            # The search feeds arbitrary strings to `re`, which warns about some of them on stderr.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return run(*args)

        options = {"gated": gated, "cost": cost, "helper": helper, "kind": "knowledge" if knowledge else "generic"}
        {1: lib.unary, 2: lib.binary, 3: lib.ternary}[arity](name, fn, template, keywords, **options)

    def text(x, low=0, high=600):
        # The search drops values with a longer repr, so longer strings can only be original arguments.
        return isinstance(x, str) and low <= len(x) <= high

    def nonempty(x):
        return text(x, 1)

    def sentence(x):
        return text(x, 1) and len(x.split()) > 0

    def phrase(x):
        return text(x, 3) and len(x.split()) > 1

    def pair(x, y):
        return text(x, 1) and text(y, 1)

    def inside(x, y):
        return pair(x, y) and y in x

    def cut(x, n):
        return text(x, 2) and is_int(n) and 0 < n < len(x)

    def binary_digits(x):
        return text(x, 1, 64) and set(x) <= {"0", "1"}

    # ------------------------------------------------------------ character codes
    add(1, "txt_ord", lambda x: text(x, 1, 1), "ord({0})", "ascii|unicode|ordinal|code point")
    add(1, "txt_ord_each", lambda x: text(x, 1, 80), "[ord(c) for c in {0}]", "ascii|unicode|ordinal|code point")

    # ------------------------------------------------------------ character classes
    # The core library filters and counts upper case, digits, letters and vowels; these are the missing classes.
    add(1, "txt_lower_only", text, "''.join(c for c in {0} if c.islower())", "lower")
    add(1, "txt_lower_count", text, "sum(c.islower() for c in {0})",
        "count&lower|number of&lower|how many&lower|count&lower&letter|count&lower&char")
    add(1, "txt_consonant_count", text, "sum(c.isalpha() and c.lower() not in 'aeiou' for c in {0})", "consonant")
    # Tasks ask for several class counts at once; the example fixes the order of the tuple.
    add(1, "txt_alpha_digit_counts", nonempty, "(sum(c.isalpha() for c in {0}), sum(c.isdigit() for c in {0}))",
        "digit&letter|digit&alphabet")
    add(1, "txt_digit_alpha_counts", nonempty, "(sum(c.isdigit() for c in {0}), sum(c.isalpha() for c in {0}))",
        "digit&letter|digit&alphabet")
    add(1, "txt_alpha_digit_special_counts", nonempty,
        "(sum(c.isalpha() for c in {0}), sum(c.isdigit() for c in {0}), sum(not c.isalnum() for c in {0}))",
        "alphabet&digit&special|letter&digit&special")
    add(1, "txt_no_punctuation", text, "''.join(c for c in {0} if c not in string.punctuation)", "punctuation")

    # ------------------------------------------------------------ case and trimming
    add(1, "txt_capitalize", nonempty, "{0}.capitalize()", "capital")
    add(1, "txt_capitalize_first_last", sentence,
        "' '.join(w[:-1] + w[-1].upper() for w in {0}.title().split())", "capital&first&last")
    add(1, "txt_lstrip", lambda x: text(x, 1) and x[0].isspace(), "{0}.lstrip()",
        "leading|lstrip|left&space|left&whitespace|beginning&space|start&space")
    add(1, "txt_rstrip", lambda x: text(x, 1) and x[-1].isspace(), "{0}.rstrip()",
        "trailing|rstrip|right&space|right&whitespace|end&space")

    # ------------------------------------------------------------ splitting
    add(1, "txt_split_at_upper", lambda x: text(x, 1) and any(c.isupper() for c in x),
        r"re.findall(r'[A-Z][^A-Z]*', {0})", "split&upper|split&capital|break&upper|break&capital")
    add(1, "txt_split_at_lower", lambda x: text(x, 1) and any(c.islower() for c in x),
        r"re.findall(r'[a-z][^a-z]*', {0})", "split&lower|break&lower")
    add(2, "txt_chunks", cut, "[{0}[i:i + {1}] for i in range(0, len({0}), {1})]",
        "chunk|split&every|parts of|pieces| n char")

    # ------------------------------------------------------------ words
    # split -> edit -> join is three steps for the search, so the common word edits are single operators.
    add(1, "txt_dedupe_words", phrase, "' '.join(dict.fromkeys({0}.split()))",
        "duplicate&word|repeated&word|unique&word")
    add(1, "txt_sort_words", phrase, "' '.join(sorted({0}.split()))", "sort&word|alphabetical&word|order&word")
    add(1, "txt_word_lengths", sentence, "[len(w) for w in {0}.split()]", "length&word")
    add(1, "txt_last_word_length", text, "len(({0}.split() or [''])[-1])", "length&last word")
    # Words are read as runs of word characters, so punctuation next to a word does not make a new word.
    add(2, "txt_most_common_words", lambda x, n: sentence(x) and is_int(n) and 0 < n <= 50,
        r"collections.Counter(re.findall(r'\w+', {0})).most_common({1})",
        "most common&word|most frequent&word|most occur&word")
    add(2, "txt_most_common_n",
        lambda x, n: (isinstance(x, (str, list, tuple)) and 0 < len(x) <= 2000 and is_int(n) and 0 < n <= 50),
        "collections.Counter({0}).most_common({1})",
        "most common|most frequent|most occur|maximum occur|highest frequen")

    # ------------------------------------------------------------ numbers written in a string
    # The core `numbers_in` reads unsigned integers only; the guard keeps this one off strings it reads the same.
    add(1, "txt_signed_ints", lambda x: text(x, 2, 300) and "-" in x, r"[int(v) for v in re.findall(r'-?\d+', {0})]",
        "integer|negative")
    add(1, "txt_floats", lambda x: text(x, 1, 300) and "." in x,
        r"[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?', {0})]", "float|decimal|real number")
    add(1, "txt_parse_float", lambda x: text(x, 1, 30) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", x) is not None,
        "float({0})", "float|decimal|real number", cost=0.5)
    add(1, "txt_roman_to_int", lambda x: text(x, 1, 30) and set(x) <= set("IVXLCDM"), "_txt_roman_to_int({0})",
        "roman", knowledge=True, helper=ROMAN)
    # The Demlo number of a run of n ones counts up to n and back down; only the length of the input matters.
    add(1, "txt_demlo", lambda x: text(x, 1, 40) and x.isdigit(),
        "''.join(map(str, list(range(1, len({0}) + 1)) + list(range(len({0}) - 1, 0, -1))))", "demlo",
        knowledge=True)

    # ------------------------------------------------------------ searching
    # Unlike the core `span`, the needle is a pattern and a miss gives None, which is what such tasks specify.
    add(2, "txt_regex_span", lambda x, y: text(x) and text(y, 1, 80),
        "(lambda m: (m.group(0), m.start(), m.end()) if m else None)(re.search({1}, {0}))",
        "regex|regular expression|pattern|match|occurr&position|substring&position", quiet=True)
    add(2, "txt_rfind", inside, "{0}.rfind({1})", "last&occurr|last&index|last&position|rightmost")
    # Positions counted from one, with None for a miss, are as common in tasks as Python's zero-based index.
    add(2, "txt_last_position", pair, "({0}.rfind({1}) + 1) or None", "last&occurr|last&position")
    add(2, "txt_char_at", lambda x, n: text(x, 1) and is_int(n) and -len(x) <= n < len(x), "{0}[{1}]",
        "character&index|character&position|char&index|kth char|nth char|k-th char|n-th char|k'th char|n'th char")
    add(1, "txt_first_unique_char", nonempty, "next((c for c in {0} if {0}.count(c) == 1), None)",
        "non-repeat|non repeat|not repeat|nonrepeat|non-recurr|first&unique&char|first&once", knowledge=True)
    add(1, "txt_first_repeated_char", nonempty, "next((c for i, c in enumerate({0}) if c in {0}[:i]), None)",
        "first&repeated|first&repeating|first&duplicate|first&recurr", knowledge=True)

    # ------------------------------------------------------------ edits
    add(1, "txt_reverse_vowels", nonempty, "_txt_reverse_vowels({0})", "revers&vowel", knowledge=True,
        helper=REVERSE_VOWELS)
    add(1, "txt_squeeze_repeats", lambda x: text(x, 2), "''.join(k for k, _ in itertools.groupby({0}))",
        "consecutive&duplicate|adjacent&duplicate|consecutive&repeat|consecutive&same|adjacent&same")
    add(1, "txt_dedupe_chars", lambda x: text(x, 2), "''.join(dict.fromkeys({0}))",
        "duplicate&char|duplicate&letter|repeated&char|repeated&letter|unique&char")
    add(2, "txt_squeeze_char", lambda x, y: text(x, 2) and text(y, 1, 1) and y * 2 in x,
        "re.sub('(?:' + re.escape({1}) + '){{2,}}', lambda m: {1}, {0})",
        "multiple&single|consecutive&occur|repeated&single|squeeze|collapse")
    # The core rotations take lists only.
    add(2, "txt_rotate_left", cut, "{0}[{1}:] + {0}[:{1}]", "rotat")
    add(2, "txt_rotate_right", cut, "{0}[-{1}:] + {0}[:-{1}]", "rotat")

    # ------------------------------------------------------------ subsequence definitions
    def short(*values):
        return all(isinstance(v, (str, list, tuple)) and 0 < len(v) <= 60 for v in values)

    add(3, "txt_lcs3_length", lambda x, y, z: short(x, y, z) and max(len(x), len(y), len(z)) <= 30,
        "_txt_lcs3({0}, {1}, {2})", "longest common subsequence&three", knowledge=True, helper=LCS3)
    add(2, "txt_lcs_length", short, "_txt_lcs({0}, {1})", "longest common subsequence|lcs", knowledge=True,
        helper=LCS)
    add(2, "txt_scs_length", short, "len({0}) + len({1}) - _txt_lcs({0}, {1})",
        "supersequence|super-sequence|shortest&subsequence", knowledge=True, helper=LCS)
    add(1, "txt_repeating_subsequence_length", lambda x: text(x, 1, 60), "_txt_repeating_subsequence({0})",
        "repeating subseq|repeated subseq", knowledge=True, helper=REPEATING)

    # ------------------------------------------------------------ counting definitions
    # A substring with equal ends is a choice of two equal characters, or of one character twice.
    add(1, "txt_equal_ends_substrings", lambda x: text(x, 1, 300),
        "sum(n * (n + 1) // 2 for n in collections.Counter({0}).values())",
        "substring&start&end&same|substring&first&last&same|substring&equal&end|substring&same&end", knowledge=True)
    add(1, "txt_min_flips_alternate", binary_digits,
        "min(sum(c != '01'[i % 2] for i, c in enumerate({0})), sum(c != '10'[i % 2] for i, c in enumerate({0})))",
        "flip&alternat", knowledge=True)
    add(1, "txt_bracket_swaps", lambda x: text(x, 2, 200) and set(x) <= {"[", "]"}, "_txt_bracket_swaps({0})",
        "swap&bracket", knowledge=True, helper=BRACKET_SWAPS)
    # Two letters fit every way of ordering the permutations, so only a longer example may select this order.
    add(1, "txt_permutations", lambda x: text(x, 3, 5), "[''.join(p) for p in itertools.permutations({0})]",
        "permutation")

    # ------------------------------------------------------------ named conventions
    # An adverb is taken to be a word ending in "ly", as in the `re` documentation example these tasks come from.
    add(1, "txt_adverb_span", text,
        r"next(((m.start(), m.end(), m.group(0)) for m in re.finditer(r'\w+ly', {0})), None)", "adverb",
        knowledge=True)
    add(1, "txt_adverb_text", text,
        r"next(('%d-%d: %s' % (m.start(), m.end(), m.group(0)) for m in re.finditer(r'\w+ly', {0})), None)",
        "adverb", knowledge=True)
    # The usual one-line clean-up: zeros are dropped after each dot, the first group is left as it is.
    add(1, "txt_ip_strip_zeros", lambda x: text(x, 7, 60) and x.count(".") == 3, r"re.sub(r'\.0*', '.', {0})",
        "ip address&zero|ip-address&zero|ipv4&zero", knowledge=True)
    # A date inside a URL path is /yyyy/mm/dd/; other digit runs of a link (ids, ports) must not be read as one.
    add(1, "txt_url_dates", lambda x: text(x, 8),
        r"re.findall(r'/(\d{{4}})/(\d{{1,2}})/(\d{{1,2}})/', {0})", "year&month|date&url", knowledge=True)
    # A fifth of the unsolved string tasks are predicates that answer in words. The predicate belongs to
    # another pack; these only put its truth value into the usual pair of words, so they accept bools alone
    # and the gate can be as wide as "this task asks a yes/no question".
    for name, yes, no in [("match_message", "Found a match!", "Not matched!"), ("yes_no", "Yes", "No"),
                          ("valid_invalid", "Valid", "Invalid")]:
        add(1, "txt_" + name, lambda x: isinstance(x, bool), f"{yes!r} if {{0}} else {no!r}",
            "check|whether|match|regex|sequence|valid")

    # "Remove even characters" counts positions from one, so it keeps what Python calls the even indices.
    lib.also_named("even_index/1", "even char")
