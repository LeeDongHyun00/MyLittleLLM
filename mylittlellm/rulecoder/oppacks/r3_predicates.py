"""Round 3 operators for predicates: gated True/False properties of numbers, strings, sequences and pairs.

The core library has no operator that returns a bool, so every task whose example expects True or
False needs one of these. A single True/False example is reproduced by half of all predicates, which
is why every operator here is gated on words that name its property and guarded on the argument
types the property is defined for. When two named predicates both reproduce the example, the engine
takes the one found first, or with "specific_first" the one named by the longer keyword. Both
tie-breaks are served the same way: the more specific reading is registered first and carries the
longer wording.
"""
import itertools
import math
import re
import string

# Multi-line definitions are kept as source text because the same text is pasted into generated answers.
KEITH = '''def _pred_keith(n):
    if n < 10:
        return False
    terms = [int(d) for d in str(n)]
    size = len(terms)
    # Each new term is the sum of the previous `size` terms; n is Keith when the sequence hits it.
    while terms[-1] < n:
        terms.append(sum(terms[-size:]))
    return terms[-1] == n
'''
ALIQUOT = '''def _pred_aliquot(n):
    # Sum of the proper divisors, pairing d with n // d so that large n stay cheap.
    total = 0
    for d in range(1, math.isqrt(n) + 1):
        if n % d == 0:
            total += d if d * d == n else d + n // d
    return total - n
'''
HAPPY = '''def _pred_happy(n):
    seen = set()
    # Replace n by the sum of its squared digits until it reaches 1 or repeats.
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(d) ** 2 for d in str(n))
    return n == 1
'''
FIBONACCI = '''def _pred_fibonacci(n):
    a, b = 0, 1
    while a < n:
        a, b = b, a + b
    return a == n
'''
POWER_OF = '''def _pred_power_of(n, k):
    if n < 1 or k < 2:
        return False
    while n % k == 0:
        n //= k
    return n == 1
'''
BALANCED = '''def _pred_balanced(text):
    # An empty expression counts as not balanced: the hidden tests of the task seen expect that.
    if len(text) == 0:
        return False
    stack = []
    for ch in text:
        if ch in '([{':
            stack.append(ch)
        elif ch in ')]}':
            if not stack or stack.pop() + ch not in ('()', '[]', '{}'):
                return False
    return not stack
'''
MAGIC_SQUARE = '''def _pred_magic_square(m):
    n, s = len(m), sum(m[0])
    return (all(sum(row) == s for row in m) and all(sum(col) == s for col in zip(*m))
            and sum(m[i][i] for i in range(n)) == s and sum(m[i][n - 1 - i] for i in range(n)) == s)
'''
VALID_DATE = '''def _pred_valid_date(month, day, year):
    try:
        month, day, year = int(month), int(day), int(year)
    except ValueError:
        return False
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return 1 <= year <= 9999 and 1 <= month <= 12 and 1 <= day <= days[month - 1]
'''
SUBSET_SUM = '''def _pred_subset_sum(nums, target):
    sums = {0}
    for v in nums:
        sums |= {s + v for s in sums}
    return target in sums
'''
SUBSET_DIVISIBLE = '''def _pred_subset_divisible(nums, m):
    # Remainders reachable by a non-empty subset; at most m of them, whatever the list length.
    seen = set()
    for v in nums:
        seen |= {(r + v) % m for r in seen} | {v % m}
        if 0 in seen:
            return True
    return False
'''

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")


def register(lib):
    need, is_int, is_num, is_seq = lib.need, lib.is_int, lib.is_num, lib.is_seq
    numseq, intseq, seqseq, strseq = lib.numseq, lib.intseq, lib.seqseq, lib.strseq
    scope = {"math": math, "re": re, "itertools": itertools}
    compiled = {}

    def predicate(arity, name, guard, template, keywords, helper=None):
        # The search function is compiled from the emitted template, so the program the engine
        # verifies on the example is by construction the program it ships.
        if helper and helper not in compiled:
            compiled[helper] = lib.helper(helper)[0]
            scope[compiled[helper].__name__] = compiled[helper]
        names = "abc"[:arity]
        body = eval("lambda " + ", ".join(names) + ": " + template.format(*names), scope)

        def run(*args):
            need(guard(*args))
            result = body(*args)
            need(result is True or result is False)     # 0/1 or None would never equal a bool target
            return result
        lib.know(arity, name, run, template, keywords, helper=helper)

    def is_str(x):
        return isinstance(x, str)

    def sized(x):
        return isinstance(x, (str, list, tuple, dict, set))

    def natural(limit):
        return lambda n: is_int(n) and 0 <= n <= limit

    def ints(a, b):
        return is_int(a) and is_int(b)

    def strs(a, b):
        return is_str(a) and is_str(b) and len(a) > 0 and len(b) > 0

    def seqs(a, b):
        # A list and its own tuple(...) copy are not two sequences: the search pairs every argument
        # with its conversions, and a containment test of a sequence with itself is always true.
        return (is_seq(a) and is_seq(b) and len(a) > 0 and len(b) > 0
                and not (type(a) is not type(b) and list(a) == list(b)))

    # ------------------------------------------------------------ integers: named classes of numbers
    # The isinstance test is part of the definition: a Woodall number is an integer n * 2**n - 1.
    predicate(1, "is_woodall", lambda n: is_int(n) and abs(n) < 10 ** 30,
              "isinstance({0}, int) and any(k * 2 ** k - 1 == {0} for k in range(1, {0}.bit_length() + 1))",
              "woodall|woodball")
    predicate(1, "is_keith", natural(10 ** 12), "_pred_keith({0})", "keith", helper=KEITH)
    predicate(1, "is_armstrong", natural(10 ** 18),
              "{0} == sum(int(d) ** len(str({0})) for d in str({0}))", "armstrong|narcissistic")
    predicate(1, "is_abundant", natural(10 ** 8), "{0} > 0 and _pred_aliquot({0}) > {0}", "abundant",
              helper=ALIQUOT)
    predicate(1, "is_deficient", natural(10 ** 8), "{0} > 0 and _pred_aliquot({0}) < {0}", "deficient",
              helper=ALIQUOT)
    predicate(1, "is_perfect_number", natural(10 ** 8), "{0} > 0 and _pred_aliquot({0}) == {0}",
              "perfect number", helper=ALIQUOT)
    predicate(1, "is_harshad", natural(10 ** 18), "{0} > 0 and {0} % sum(int(d) for d in str({0})) == 0",
              "harshad|niven")
    predicate(1, "is_happy", natural(10 ** 18), "{0} > 0 and _pred_happy({0})", "happy number|is happy",
              helper=HAPPY)
    predicate(1, "is_fibonacci_number", natural(10 ** 30), "_pred_fibonacci({0})",
              "is a fibonacci|is fibonacci|fibonacci number or not|fibonacci or not", helper=FIBONACCI)
    predicate(1, "is_perfect_square", lambda n: is_int(n) and abs(n) < 10 ** 30,
              "{0} >= 0 and math.isqrt({0}) ** 2 == {0}", "perfect square")
    predicate(1, "is_power_of_two", is_int, "{0} > 0 and {0} & ({0} - 1) == 0",
              "a power of two|a power of 2|is power of two|is power of 2|power of two or not|power of 2 or not")
    predicate(1, "is_leap_year", lambda n: is_int(n) and 0 < n < 10000,
              "{0} % 4 == 0 and ({0} % 100 != 0 or {0} % 400 == 0)", "leap year|leap-year|leap or not")
    # "prime" is matched with its left neighbour so that co-prime and non-prime do not name this rule.
    predicate(1, "not_prime", lambda n: is_int(n) and abs(n) <= 10 ** 9,
              "{0} < 2 or any({0} % d == 0 for d in range(2, int(math.sqrt({0})) + 1))",
              "non-prime|nonprime|not prime|not a prime")
    predicate(1, "is_prime", lambda n: is_int(n) and abs(n) <= 10 ** 9,
              "{0} > 1 and all({0} % d for d in range(2, int(math.sqrt({0})) + 1))",
              "is prime|is a prime| prime number| prime or not|primality")
    predicate(1, "is_palindrome_number", natural(10 ** 30), "str({0}) == str({0})[::-1]", "palindrom")
    # Calendar knowledge, once for month numbers and once for month names; the argument type decides.
    month_number = lambda n: is_int(n) and 1 <= n <= 12  # noqa: E731
    month_name = lambda s: is_str(s) and s in MONTHS  # noqa: E731
    for days, numbers in ((31, (1, 3, 5, 7, 8, 10, 12)), (30, (4, 6, 9, 11)), (28, (2,))):
        names = tuple(MONTHS[i - 1] for i in numbers)
        number_test = "{0} == 2" if days == 28 else f"{{0}} in {numbers!r}"
        name_test = "{0} == 'February'" if days == 28 else f"{{0}} in {names!r}"
        predicate(1, f"month_number_has_{days}_days", month_number, number_test, f"month&{days} day")
        predicate(1, f"month_name_has_{days}_days", month_name, name_test, f"month&{days} day")
    # The frozen engine cannot take a constant out of the description, so the divisibility test by
    # a small constant is one gated rule per divisor (the two-argument form is `divisible` below).
    # The number is matched with the character after it: "divisible by 2" must not name 25.
    def by(k):
        return "|".join(f"{phrase} {k}{end}" for phrase in ("divisible by", "multiple of") for end in " .,?)")

    for k in range(3, 14):
        predicate(1, f"divisible_by_{k}", is_int, f"{{0}} % {k} == 0", by(k))
    # "is even" is matched with what follows it, so that "is evenly divisible" does not name the rule.
    predicate(1, "is_odd", is_num, "{0} % 2 != 0",
              "is odd |is odd.|is odd,|is an odd|odd or not|odd parity|parity&odd|odd number or")
    predicate(1, "is_even", is_num, "{0} % 2 == 0",
              "is even |is even.|is even,|is an even|even or not|even parity|parity&even|even number or|"
              + by(2))
    # "Parity" also means the parity of the number of set bits. That reading comes second: it is
    # taken only when the example contradicts the plain odd/even reading.
    predicate(1, "has_odd_bit_parity", natural(2 ** 64), "bin({0}).count('1') % 2 == 1",
              "odd parity|parity&odd")
    predicate(1, "has_even_bit_parity", natural(2 ** 64), "bin({0}).count('1') % 2 == 0",
              "even parity|parity&even")

    # ------------------------------------------------------------ strings
    predicate(1, "is_palindrome", lambda x: isinstance(x, (str, list, tuple)) and len(x) > 0,
              "{0} == {0}[::-1]", "palindrom")
    predicate(1, "is_balanced", is_str, "_pred_balanced({0})",
              "balanced|valid&parenthes|parenthes&match|valid&bracket", helper=BALANCED)
    predicate(1, "is_decimal_2", is_str, "re.search(r'^[0-9]+(\\.[0-9]{{1,2}})?$', {0}) is not None",
              "decimal&precision of 2|decimal&2 decimal|decimal&two decimal")
    predicate(1, "has_letter_and_digit", is_str,
              "any(c.isalpha() for c in {0}) and any(c.isdigit() for c in {0})",
              "one letter&one number|one letter&one digit|letter&number&least|letter&digit&least")
    predicate(1, "is_alphanumeric", is_str, "{0}.isalnum()", "alphanumeric|a-z&0-9")
    predicate(1, "is_pangram", is_str, "set('abcdefghijklmnopqrstuvwxyz') <= set({0}.lower())", "pangram")
    predicate(1, "is_digits_only", is_str, "{0}.isdigit()",
              "only digits|digits only|only of digits|all digits|only numeric|is numeric")
    predicate(1, "is_letters_only", is_str, "{0}.isalpha()",
              "only letters|letters only|only alphabet|alphabets only|only of letters|all letters")
    predicate(1, "is_uppercase", is_str, "{0}.isupper()",
              "is uppercase|is upper case|in uppercase|in upper case|all uppercase|all upper case"
              "|uppercase or not|all capital")
    predicate(1, "is_lowercase", is_str, "{0}.islower()",
              "is lowercase|is lower case|in lowercase|in lower case|all lowercase|all lower case"
              "|lowercase or not")
    predicate(1, "is_integer_string", is_str, "re.fullmatch(r'[+-]?[0-9]+', {0}.strip()) is not None",
              "represents an integer|represent an integer|is an integer|valid integer|integer or not")
    predicate(1, "starts_with_digit", is_str, "{0}[:1].isdigit()",
              "starts with a number|start with a number|starts with a digit|start with a digit"
              "|begins with a number|begins with a digit|number at the start|number at the beginning"
              "|digit at the start|digit at the beginning")
    predicate(1, "ends_with_digit", is_str, "{0}[-1:].isdigit()",
              "number at the end|digit at the end|ends with a number|end with a number|ends with a digit"
              "|end with a digit")
    predicate(1, "starts_with_vowel", is_str, "{0}[:1].lower() in ('a', 'e', 'i', 'o', 'u')",
              "starts with a vowel|start with a vowel|starting with a vowel|starting with vowel"
              "|begins with a vowel|starts with vowel")
    predicate(1, "is_vowel", lambda c: is_str(c) and len(c) == 1, "{0}.lower() in ('a', 'e', 'i', 'o', 'u')",
              "is a vowel|is vowel|vowel or not|vowel or consonant")
    predicate(1, "same_first_last", lambda x: isinstance(x, (str, list, tuple)) and len(x) > 0,
              "len({0}) > 0 and {0}[0] == {0}[-1]",
              "starts and ends with the same|start and end with the same|begins and ends with the same"
              "|first and last&same|first and last&equal")
    predicate(1, "two_alternating", lambda s: is_str(s) and len(s) > 1,
              "len({0}) > 1 and {0}[0] != {0}[1] and all({0}[i] == {0}[i + 2] for i in range(len({0}) - 2))",
              "two alternating|alternating characters")
    predicate(1, "is_valid_url", is_str,
              "re.search(r'(http|https)://(www\\.)?[a-zA-Z0-9@:%._+~#?&/=]{{2,256}}\\.[a-z]{{2,6}}\\b', {0})"
              " is not None", " url")

    # ------------------------------------------------------------ one sequence
    seq_or_str = lambda x: (is_seq(x) or is_str(x)) and len(x) > 0  # noqa: E731
    # Order properties need at least two items: a one-item slice of the argument is "sorted" in
    # every sense, which would let the search fit a True example with it.
    several = lambda x: is_seq(x) and len(x) > 1  # noqa: E731
    predicate(1, "has_duplicates", seq_or_str, "len({0}) != len(set({0}))",
              "contain&duplicate|any duplicate|has duplicate|have duplicate|has any duplicate|duplicates or not"
              "|duplicate or not|contain&repeated|has repeated|have repeated|any repeated")
    predicate(1, "all_unique", seq_or_str, "len({0}) == len(set({0}))",
              "are unique|is unique|unique or not|all unique| all &unique| all &distinct| all &different"
              "|no duplicate|isogram")
    # Named with "all": two sequences compared with each other are the two-argument rule same_length.
    # It stands before all_same, which the words "all ... same length" name as well.
    predicate(1, "same_lengths", lambda x: is_seq(x) and len(x) > 0 and all(sized(v) for v in x),
              "len(set(len(v) for v in {0})) == 1",
              " all &the same length| all &equal length| all &same length| all &equal size| all &same size")
    # "equal" names this rule only in the forms that cannot continue with "to": a list compared
    # with a given value is the two-argument rule all_equal_to.
    predicate(1, "all_same", seq_or_str, "all(v == {0}[0] for v in {0})",
              "only one distinct|one distinct element| all &are same| all &are the same| all &is the same"
              "| all &same or not|all same| all &identical| all &equal or not| all &are equal.|all equal"
              "|all the same|same element|same value|every element&same|every item&same"
              "|every character&same")
    predicate(1, "strictly_increasing", several, "all(a < b for a, b in zip({0}, {0}[1:]))",
              "strictly increasing|strictly ascending")
    predicate(1, "strictly_decreasing", several, "all(a > b for a, b in zip({0}, {0}[1:]))",
              "strictly decreasing|strictly descending")
    predicate(1, "is_sorted", several, "all(a <= b for a, b in zip({0}, {0}[1:]))",
              "is sorted|are sorted|sorted or not|sorted in ascending|in ascending order|in increasing order"
              "|in sorted order|non-decreasing or not|is non-decreasing|non-decreasing|increasing trend"
              "|is increasing| increasing or not")
    predicate(1, "is_sorted_descending", several, "all(a >= b for a, b in zip({0}, {0}[1:]))",
              "sorted in descending|in descending order|in decreasing order|non-increasing or not"
              "|is non-increasing|non-increasing|is decreasing| decreasing or not")
    predicate(1, "is_monotonic", several,
              "all(a <= b for a, b in zip({0}, {0}[1:])) or all(a >= b for a, b in zip({0}, {0}[1:]))",
              "monotonic|monotone")
    predicate(1, "has_none", lambda x: is_seq(x) and len(x) > 0, "any(v is None for v in {0})",
              "none value|any none|has none|have none|contains none|contain none|none element|none or not"
              "|is none or|are none")
    predicate(1, "is_min_heap", numseq, "all({0}[(i - 1) // 2] <= {0}[i] for i in range(1, len({0})))",
              "min heap|min-heap|minheap")
    predicate(1, "is_max_heap", numseq, "all({0}[(i - 1) // 2] >= {0}[i] for i in range(1, len({0})))",
              "max heap|max-heap|maxheap")
    predicate(1, "is_magic_square",
              lambda m: seqseq(m) and all(len(row) == len(m) and numseq(row) for row in m),
              "_pred_magic_square({0})", "magic square", helper=MAGIC_SQUARE)
    # "Even index holds an even number" is meant both ways in the one task seen (odd index, odd
    # number as well), so the parity-match reading is tried before the literal one-sided readings.
    even_words = "even ind&even number|even position&even number"
    odd_words = "odd ind&odd number|odd position&odd number"
    predicate(1, "index_parity_match", intseq, "all(v % 2 == i % 2 for i, v in enumerate({0}))",
              even_words + "|" + odd_words)
    predicate(1, "even_at_even_index", intseq, "all(v % 2 == 0 for v in {0}[::2])", even_words)
    predicate(1, "odd_at_odd_index", intseq, "all(v % 2 == 1 for v in {0}[1::2])", odd_words)
    predicate(1, "all_truthy", lambda x: is_seq(x) and len(x) > 0 and all(isinstance(v, bool) for v in x),
              "all({0})", "all valid|all true|all&are true|valid values")
    predicate(1, "all_empty", lambda x: is_seq(x) and len(x) > 0 and all(sized(v) for v in x),
              "all(not v for v in {0})",
              " all &empty or not| all &are empty| all &is empty| all &empty|every&is empty|every&empty")
    predicate(1, "is_empty", sized, "len({0}) == 0", "is empty|are empty|empty or not")

    # ------------------------------------------------------------ two arguments
    # A relation between two arguments is named by wording that announces the second one ("subset
    # of another", "the given sublist", "of each other"). With a single argument the search pairs it
    # with its own slices and rearrangements, for which containment and equal length always hold.
    predicate(2, "is_sublist", lambda x, y: seqs(x, y) and type(x) is type(y),
              "any({0}[i:i + len({1})] == {1} for i in range(len({0}) - len({1}) + 1))",
              "given sublist|is a sublist|is sublist|sublist of|sub list of|given subarray|is a subarray"
              "|is subarray|subarray of|sub array of")
    # A row never equals a scalar, so rows compared with a scalar belong to the nested rule below.
    predicate(2, "all_equal_to", lambda x, v: is_seq(x) and len(x) > 0 and (is_seq(v) or not seqseq(x)),
              "all(e == {1} for e in {0})", "are equal to|all equal to")
    predicate(2, "all_nested_equal_to", lambda x, v: seqseq(x) and not is_seq(v),
              "all(e == {1} for row in {0} for e in row)", "all k elements|are equal to|all equal to")
    predicate(2, "all_values_equal_to", lambda d, v: isinstance(d, dict) and len(d) > 0,
              "all(e == {1} for e in {0}.values())", "all values|values are same|values are equal")
    # Named only when the wording announces a second argument. "Starts with a vowel" has one
    # argument, and there the search would pair the string with its own first character, which
    # is true of every string. Both come before has_substring, which holds whenever they do.
    # A quote after the verb announces a literal, which the engine may supply as the second operand.
    second = "the given|a given|the specified|a specified|another|'|\""
    predicate(2, "starts_with", strs, "{0}.startswith({1})",
              "|".join(f"{verb} {what}" for verb in ("starts with", "start with", "begins with")
                       for what in second.split("|")) + "|prefix")
    predicate(2, "ends_with", strs, "{0}.endswith({1})",
              "|".join(f"{verb} {what}" for verb in ("ends with", "end with", "ending with", "ending in")
                       for what in second.split("|")) + "|suffix")
    predicate(2, "has_substring", strs, "{1} in {0}",
              "substring|sub-string|sub string|contains another string|contain another string|contains the string"
              "|contains a string|contains the word|contains a word|contains '|containing '|contain '"
              "|contains \"|containing \"|contain \"")
    predicate(2, "any_has_substring", lambda x, t: strseq(x) and is_str(t) and len(t) > 0,
              "any({1} in s for s in {0})", "substring|sub-string|sub string")
    # A flat list never holds another list, so a sequence is accepted as the element only when the
    # container is nested; this also keeps the rule away from sublist and subset tasks. A string in
    # the one-item list of itself is left out: that is what the search builds from a word and its split().
    predicate(2, "contains",
              lambda x, v: isinstance(x, (list, tuple, set, dict)) and len(x) > 0
              and not isinstance(v, (set, dict))
              and (not is_seq(v) or (is_seq(x) and any(is_seq(e) for e in x)))
              and not (is_str(v) and is_seq(x) and len(x) == 1 and x[0] == v),
              "{1} in {0}",
              "exists in|exists within|exist in|exist within|contains the|contain the|contains a given"
              "|contain a given|contains an element|contain an element|contains a value|contains an item"
              "|is present|present in|search&element|is in the|member of|includes the")
    # "The number is greater than every element" and "every element is less than the number" are
    # the same test; the wording decides which of the two directions is meant.
    predicate(2, "all_below", lambda x, n: numseq(x) and is_num(n), "all({1} > v for v in {0})",
              "greater than the elements|greater than all|greater than every|greater than each"
              "|values&less than|values&smaller than|elements&are less than|elements&are smaller than"
              "|all&are less than|all&are smaller than")
    predicate(2, "all_above", lambda x, n: numseq(x) and is_num(n), "all({1} < v for v in {0})",
              "smaller than the elements|less than the elements|smaller than all|less than all"
              "|smaller than every|less than every|smaller than each|less than each|values&greater than"
              "|elements&are greater than|all&are greater than")
    pairwise = lambda x, y: seqs(x, y) and numseq(x) and numseq(y) and len(x) == len(y)  # noqa: E731
    predicate(2, "each_greater", pairwise, "all(a > b for a, b in zip({0}, {1}))",
              "corresponding&greater|corresponding&larger|corresponding&smaller|corresponding&less")
    predicate(2, "each_less", pairwise, "all(a < b for a, b in zip({0}, {1}))",
              "corresponding&greater|corresponding&larger|corresponding&smaller|corresponding&less")
    predicate(2, "differ_one_bit", ints, "({0} ^ {1}) > 0 and ({0} ^ {1}) & (({0} ^ {1}) - 1) == 0",
              "differ&one bit|differ&bit position|differ&single bit")
    # Every item of the second argument is always paired with the same item of the first.
    predicate(2, "follows_pattern", lambda x, y: seqs(x, y) or strs(x, y),
              "len({0}) == len({1}) and len(set(zip({0}, {1}))) == len(set({1}))",
              "pattern&follow|same pattern|pattern&sequence")
    predicate(2, "is_isomorphic", strs,
              "len({0}) == len({1}) and len(set(zip({0}, {1}))) == len(set({0})) == len(set({1}))",
              "isomorphic")
    predicate(2, "contains_all", seqs, "all(v in {0} for v in {1})",
              "subset of another|subset of other|subset of the other|subset of the second|subset of the first"
              "|is subset of|superset|contains all|contain all")
    predicate(2, "only_from", seqs, "all(v in {1} for v in {0})",
              "contains only|contain only|consists only|consist only|all elements&present in"
              "|all the elements&present in")
    predicate(2, "any_common", seqs, "any(v in {1} for v in {0})",
              "at least one common|any common|one common element|any&present in|any&element&present"
              "|have common|has common|share")
    predicate(2, "are_disjoint", seqs, "not any(v in {1} for v in {0})",
              "disjoint|no common|nothing in common|not have any common")
    predicate(2, "is_rotation", strs, "len({0}) == len({1}) and {1} in {0} + {0}",
              "rotations of each other|rotation of each other|rotation of another|rotation of the other"
              "|is a rotation of|are rotations")
    predicate(2, "is_repetition_of", strs, "len({1}) > 0 and {0} == {1} * (len({0}) // len({1}))",
              "concatenation of|repetition of|repeated copies|multiple copies")
    predicate(2, "same_digit_count", ints, "len(str(abs({0}))) == len(str(abs({1})))",
              "same number of digits|equal number of digits|same count of digits")
    predicate(2, "same_length", lambda x, y: seqs(x, y) or strs(x, y), "len({0}) == len({1})",
              "two&same length|two&equal length|both&same length|both&equal length|same length as"
              "|equal length as")
    predicate(2, "parallel_lines",
              lambda x, y: numseq(x) and numseq(y) and 2 <= len(x) <= 3 and 2 <= len(y) <= 3,
              "{0}[0] * {1}[1] == {1}[0] * {0}[1]", "parallel or not|are parallel|lines&parallel")
    # With the engine feature that reads constants from the description, "divisible by 25" is this
    # rule with 25 as second operand; the small divisors above also work without that feature.
    predicate(2, "divisible", lambda a, b: ints(a, b) and b != 0, "{0} % {1} == 0",
              "divisible by|multiple of|is a factor of|is a divisor of|divides the|evenly divide")
    predicate(2, "is_power_of", lambda n, k: ints(n, k) and 1 <= n <= 10 ** 30 and 2 <= k <= 10 ** 6,
              "_pred_power_of({0}, {1})", "a power of|is power of", helper=POWER_OF)
    predicate(2, "coprime", ints, "math.gcd({0}, {1}) == 1", "coprime|co-prime|relatively prime|mutually prime")
    predicate(2, "opposite_signs", ints, "({0} ^ {1}) < 0", "opposite sign|different sign")
    # Tasks translated from C pass the array length as an extra argument; a target equal to that
    # length is taken to be the length, not the value searched for.
    target = lambda x, n, size: intseq(x) and len(x) <= size and is_int(n) and n != len(x)  # noqa: E731
    predicate(2, "subset_sum", lambda x, n: target(x, n, 12), "_pred_subset_sum({0}, {1})",
              "subset&sum equal|subset sum|subset&given sum", helper=SUBSET_SUM)
    predicate(2, "subset_sum_divisible", lambda x, n: target(x, n, 16) and n > 0,
              "_pred_subset_divisible({0}, {1})", "subset&sum divisible|subset&divisible", helper=SUBSET_DIVISIBLE)
    predicate(2, "triplet_sum", lambda x, n: target(x, n, 16),
              "any(sum(t) == {1} for t in itertools.combinations({0}, 3))", "triplet&sum")
    # Plain equality is False for almost any pair, so it would fit most False examples: the two
    # equality rules come after every other two-argument rule and are named by narrow phrases.
    predicate(2, "identical", lambda x, y: type(x) is type(y) and sized(x), "{0} == {1}",
              "identical|strings are equal|lists are equal|tuples are equal")
    predicate(2, "same_elements", lambda x, y: seqs(x, y) or strs(x, y), "sorted({0}) == sorted({1})",
              "permutations of each other|permutation of each other|permutation of another"
              "|permutation of the other|is a permutation of|are permutations|anagram|arrays are equal"
              "|two arrays&equal")

    # ------------------------------------------------------------ three arguments
    def numbers(a, b, c):
        return is_num(a) and is_num(b) and is_num(c)

    def bit_range(n, low, high):
        return (is_int(n) and is_int(low) and is_int(high) and 0 <= n < 2 ** 64 and 1 <= low <= 64
                and 0 <= high <= 64)

    predicate(3, "is_equilateral", numbers, "{0} == {1} == {2}", "equilateral")
    predicate(3, "is_isosceles", numbers, "{0} == {1} or {1} == {2} or {0} == {2}", "isosceles")
    predicate(3, "is_scalene", numbers, "{0} != {1} and {1} != {2} and {0} != {2}", "scalene")
    # Sides before angles: a description that mentions sides names both rules.
    predicate(3, "triangle_sides_valid", numbers,
              "{0} + {1} > {2} and {0} + {2} > {1} and {1} + {2} > {0}",
              "triangle&valid&side|triangle&side&possible|triangle&side&form")
    # The space keeps the "angle" inside "triangle" from naming this rule on its own.
    predicate(3, "triangle_angles_valid", numbers,
              "{0} > 0 and {1} > 0 and {2} > 0 and {0} + {1} + {2} == 180", "triangle&valid|triangle& angle")
    predicate(3, "is_right_triangle", lambda a, b, c: numbers(a, b, c) and min(a, b, c) > 0,
              "2 * max({0}, {1}, {2}) ** 2 == {0} ** 2 + {1} ** 2 + {2} ** 2",
              "right angle|right-angle|right triangle|right angled|pythagor")
    date_part = lambda v: is_int(v) or (is_str(v) and v.isdigit())  # noqa: E731
    predicate(3, "is_valid_date", lambda m, d, y: date_part(m) and date_part(d) and date_part(y),
              "_pred_valid_date({0}, {1}, {2})", "valid& date|gregorian date", helper=VALID_DATE)
    # Bits low..high counted from 1 at the least significant end, as bit-manipulation tasks do.
    mask = "(((1 << {2}) - 1) ^ ((1 << ({1} - 1)) - 1))"
    predicate(3, "bits_unset_in_range", bit_range, "{0} & " + mask + " == 0",
              "all the bits&unset&range|bits are unset&range|bit&unset&range")
    predicate(3, "bits_set_in_range", bit_range, "{0} & " + mask + " == " + mask,
              "all the bits&range|bits&set&range|bits are set&range")
    predicate(3, "in_arithmetic_sequence", lambda a, b, c: is_int(a) and is_int(b) and is_int(c),
              "{0} == {1} or (({1} - {0}) * {2} > 0 and ({1} - {0}) % {2} == 0)", "infinite sequence")

    # ------------------------------------------------------------ a letter named in the description
    # "contains 'z'" is membership with a constant the engine cannot read from the description, so
    # there is one gated rule per quoted letter. The inside-a-word form comes first: it is named by a
    # superset of the words that name plain membership.
    for letter in string.ascii_lowercase:
        predicate(1, f"has_inner_{letter}", is_str, f"re.search(r'\\B{letter}\\B', {{0}}) is not None",
                  f"'{letter}'&except&start&end|'{letter}'&not at the start|'{letter}'&middle of")
    for letter in string.ascii_lowercase:
        quoted = f"'{letter}'", f'"{letter}"'
        predicate(1, f"has_letter_{letter}", is_str, f"'{letter}' in {{0}}",
                  "|".join(f"{verb} {q}" for verb in ("containing", "contains", "contain") for q in quoted)
                  + f"|contain&letter '{letter}'")
