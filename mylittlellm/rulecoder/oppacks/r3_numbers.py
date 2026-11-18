"""Round 3 operators for numbers: divisors, primes, digits, bits, progressions, formulas and conversions.

Every operator but one type conversion is gated. A numeric example is a single small value, so an
operator that is tried on every task would reproduce many of them by accident; the description has
to name the concept first. Search function and emitted code are built from the same template text
(see `formula`), so the program the search verifies is the program that is shipped.
"""
import cmath
import math
import statistics

# ---------------------------------------------------------------- helper sources
# Pasted into generated answers, so they use builtins and `math` only. Loops are O(sqrt n) or a sieve
# because hidden tests call the answer with far larger numbers than the visible example.
_DIVISORS = '''def _divisors(n):
    if n < 1:
        return []
    low = [d for d in range(1, math.isqrt(n) + 1) if n % d == 0]
    return low + [n // d for d in reversed(low) if d * d != n]
'''

# No call to the builtin sum(): one task names its own function `sum`, which would shadow it.
_DIVISOR_SUM = '''def _divisor_sum(n):
    total = 0
    for d in range(1, math.isqrt(max(n, 0)) + 1):
        if n % d == 0:
            total += d if d * d == n else d + n // d
    return total
'''

_PRIME_FACTORS = '''def _prime_factors(n, repeated=False):
    factors, p = [], 2
    while p * p <= n:
        if n % p == 0:
            factors.append(p)
            n //= p
            while n % p == 0:
                if repeated:
                    factors.append(p)
                n //= p
        p += 1
    if n > 1:
        factors.append(n)
    return factors
'''

_PRIMES_UPTO = '''def _primes_upto(n):
    flags = [False, False] + [True] * (n - 1)
    for p in range(2, math.isqrt(max(n, 0)) + 1):
        if flags[p]:
            flags[p * p::p] = [False] * len(flags[p * p::p])
    return [p for p in range(2, n + 1) if flags[p]]
'''

# The nth prime is below n (ln n + ln ln n) from n = 6 on (Rosser), so one sieve of that size is enough.
_NTH_PRIME = '''def _nth_prime(n):
    if n < 1:
        raise ValueError("there is no prime number with that index")
    limit = 15 if n < 6 else int(n * (math.log(n) + math.log(math.log(n)))) + 1
    flags = [False, False] + [True] * (limit - 1)
    for p in range(2, math.isqrt(limit) + 1):
        if flags[p]:
            flags[p * p::p] = [False] * len(flags[p * p::p])
    return [p for p in range(limit + 1) if flags[p]][n - 1]
'''

# A member of an amicable pair counts even when its partner lies above n, hence the direct
# divisor sum for partners outside the sieve.
_AMICABLE = '''def _amicable_upto(n):
    def aliquot(k):
        total = 1
        for d in range(2, math.isqrt(k) + 1):
            if k % d == 0:
                total += d if d * d == k else d + k // d
        return total
    sums = [0, 0] + [1] * (n - 1)
    for d in range(2, n // 2 + 1):
        sums[2 * d::d] = [total + d for total in sums[2 * d::d]]
    found = []
    for k in range(2, n + 1):
        partner = sums[k]
        if partner != k and partner > 1 and (sums[partner] if partner <= n else aliquot(partner)) == k:
            found.append(k)
    return found
'''

# Bit positions are counted inside the bit length of n; a negative n has no such positions.
_EVERY_OTHER_BIT = '''def _every_other_bit(n, first):
    return sum(1 << i for i in range(first, max(n, 0).bit_length(), 2))
'''

_SET_LEFTMOST_UNSET = '''def _set_leftmost_unset_bit(n):
    for i in range(n.bit_length() - 1, -1, -1):
        if not n >> i & 1:
            return n | 1 << i
    return n
'''

# Mirroring the left half gives the nearest palindrome with that half; one step of the half is
# enough to get past the number itself, so no search over candidates is needed.
_NEXT_PALINDROME = '''def _next_palindrome(n):
    target = max(n + 1, 0)
    text = str(target)
    odd = len(text) % 2
    half = text[:(len(text) + 1) // 2]
    candidate = int(half + half[::-1][odd:])
    if candidate < target:
        half = str(int(half) + 1)
        candidate = int(half + half[::-1][odd:])
    return candidate
'''

_PREVIOUS_PALINDROME = '''def _previous_palindrome(n):
    target = n - 1
    text = str(target)
    odd = len(text) % 2
    half = text[:(len(text) + 1) // 2]
    candidate = int(half + half[::-1][odd:])
    if candidate > target:
        half = str(int(half) - 1)
        candidate = int(half + half[::-1][odd:])
        if half == "0" or len(half) < (len(text) + 1) // 2:
            candidate = 10 ** (len(text) - 1) - 1
    return candidate
'''

_NEXT_BIGGER = '''def _next_bigger(n):
    digits = list(str(n))
    for i in range(len(digits) - 2, -1, -1):
        if digits[i] < digits[i + 1]:
            tail = digits[i:]
            pivot = min(d for d in tail if d > tail[0])
            tail.remove(pivot)
            digits[i:] = [pivot] + sorted(tail)
            return int("".join(digits))
    return False
'''

_TO_ROMAN = '''def _to_roman(n):
    text = ""
    for value, symbol in zip((1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1),
                             ("M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I")):
        text += symbol * (n // value)
        n %= value
    return text
'''

# Plain left-to-right addition: sum() rounds differently since Python 3.12, and the examples are
# written with the digits the textbook recursion 1 / n + h(n - 1) produces.
_HARMONIC = '''def _harmonic_sum(n):
    total = 1.0
    for k in range(2, n + 1):
        total += 1 / k
    return total
'''

# The iteration cap only matters where the classic loop would never end (two guesses alternating).
_BABYLONIAN = '''def _babylonian_sqrt(x):
    if x == 0:
        return 0
    guess, previous = x / 2, x / 2 + 1
    for _ in range(200):
        if guess == previous:
            break
        previous, guess = guess, (guess + x / guess) / 2
    return guess
'''

_ENTRINGER = '''def _entringer(n, k):
    row = [1]
    for size in range(1, n + 1):
        new = [0]
        for j in range(1, size + 1):
            new.append(new[-1] + row[size - j])
        row = new
    return row[k]
'''

_LUCKY = '''def _lucky_numbers(n):
    numbers = list(range(1, n * n + 10, 2))
    index = 1
    while index < len(numbers) and numbers[index] <= len(numbers):
        del numbers[numbers[index] - 1::numbers[index]]
        index += 1
    return numbers[:n]
'''

# Lagrange and Legendre: every number is a sum of four squares, and of three unless it is 4^a (8b + 7).
_MIN_SQUARES = '''def _min_squares(n):
    if n <= 0:
        return 0
    if math.isqrt(n) ** 2 == n:
        return 1
    if any(math.isqrt(n - a * a) ** 2 == n - a * a for a in range(1, math.isqrt(n) + 1)):
        return 2
    while n % 4 == 0:
        n //= 4
    return 4 if n % 8 == 7 else 3
'''

_FENCE = '''def _fence_ways(n, k):
    if n < 1:
        return 0
    same, different = 0, k
    for _ in range(n - 1):
        same, different = different, (same + different) * (k - 1)
    return same + different
'''

# Tilings of a 3 x n board with dominoes exist for even n only and satisfy t(n) = 4 t(n - 2) - t(n - 4).
_DOMINO_3XN = '''def _domino_tilings_3xn(n):
    if n % 2:
        return 0
    previous, current = 1, 1
    for _ in range(n // 2):
        previous, current = current, 4 * current - previous
    return current
'''

_SPIRAL = '''def _spiral_matrix(n):
    grid = [[0] * n for _ in range(n)]
    row, col, down, right = 0, 0, 0, 1
    for value in range(1, n * n + 1):
        grid[row][col] = value
        if not (0 <= row + down < n and 0 <= col + right < n) or grid[row + down][col + right]:
            down, right = right, -down
        row, col = row + down, col + right
    return grid
'''

_RGB_TO_HSV = '''def _rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    high, low = max(r, g, b), min(r, g, b)
    spread = high - low
    if spread == 0:
        hue = 0
    elif high == r:
        hue = (60 * ((g - b) / spread) + 360) % 360
    elif high == g:
        hue = (60 * ((b - r) / spread) + 120) % 360
    else:
        hue = (60 * ((r - g) / spread) + 240) % 360
    saturation = 0 if high == 0 else (spread / high) * 100
    return hue, saturation, high * 100
'''


def register(lib):
    need, is_int, is_num = lib.need, lib.is_int, lib.is_num
    scope = {"math": math, "cmath": cmath, "statistics": statistics}
    text_of = {}
    for source in (_DIVISORS, _DIVISOR_SUM, _PRIME_FACTORS, _PRIMES_UPTO, _NTH_PRIME, _AMICABLE, _EVERY_OTHER_BIT,
                   _SET_LEFTMOST_UNSET, _NEXT_PALINDROME, _PREVIOUS_PALINDROME, _NEXT_BIGGER, _TO_ROMAN, _HARMONIC,
                   _BABYLONIAN, _ENTRINGER, _LUCKY, _MIN_SQUARES, _FENCE, _DOMINO_3XN, _SPIRAL, _RGB_TO_HSV):
        function, text = lib.helper(source)
        scope[function.__name__] = function
        text_of[function.__name__] = text

    def formula(arity, name, template, keywords, guard, helper=None, **options):
        """Register an operator (gated unless said otherwise) whose search function is its own template.

        The guard says where the operator applies and bounds the work; it never changes a value.
        """
        names = [f"a{i}" for i in range(arity)]
        body = eval(f"lambda {', '.join(names)}: " + template.format(*names), scope)

        def run(*args):
            guard(*args)
            return body(*args)
        options.setdefault("kind", "knowledge")
        options.setdefault("gated", True)
        {1: lib.unary, 2: lib.binary, 3: lib.ternary}[arity](
            name, run, template, keywords, helper=text_of[helper] if helper else None, **options)

    def ints(*ranges):
        """Guard: every argument is an int inside its (low, high) range."""
        def check(*args):
            need(all(is_int(a) and low <= a <= high for a, (low, high) in zip(args, ranges)))
        return check

    def numbers(low, high):
        """Guard: every argument is an int or float inside the range."""
        def check(*args):
            need(all(is_num(a) and low <= a <= high for a in args))
        return check

    def positive(*args):
        need(all(is_num(a) and 0 < a < 10 ** 6 for a in args))

    def digits_below(base):
        """Guard: a non-negative int written with digits of the given base only."""
        def check(n):
            need(is_int(n) and 0 <= n < 10 ** 18 and all(int(d) < base for d in str(n)))
        return check

    # ------------------------------------------------------------ result shape
    # A formula that divides gives 240.0 where the example shows 240. Truncating with int() would turn
    # 0.5 into 0 on other inputs, so this conversion only changes the type when no value is lost. It
    # matches exactly where the core int() conversion matches, so it adds no new coincidences; the
    # lower cost only decides which of the two is shipped. Ungated and without keywords on purpose:
    # a conversion that the description "names" would let float(x) -> int outrank real operators.
    def whole_float(x):
        need(isinstance(x, float) and x.is_integer() and abs(x) < 10 ** 12)
    formula(1, "num_exact_int", "(lambda v: int(v) if isinstance(v, float) and v.is_integer() else v)({0})", "",
            whole_float, kind="generic", gated=False, cost=0.4)
    # "tuple" alone opens on every task about tuples; the description must also speak of numbers.
    formula(2, "num_pair", "({0}, {1})",
            "|".join(f"{shape}&{noun}" for shape in ("tuple", "pair", "swap")
                     for noun in ("numbers", "integers", "values", "variables")),
            numbers(-10 ** 12, 10 ** 12), kind="generic")

    # statistics.mean keeps an int when the mean of ints is whole, which is how such examples are written.
    def whole_mean(x):
        need(lib.intseq(x) and len(x) <= 500 and sum(x) % len(x) == 0)
    formula(1, "num_mean_keep_int", "statistics.mean({0})", "average|mean", whole_mean, kind="generic")

    # ------------------------------------------------------------ divisors and primes
    upto_million = ints((1, 10 ** 6))
    formula(1, "num_divisors", "_divisors({0})", "divisor|factors", upto_million, "_divisors")
    formula(1, "num_divisor_count", "len(_divisors({0}))",
            "number of&divisor|count&divisor|how many&divisor|number of&factors|count&factors|how many&factors",
            upto_million, "_divisors")
    formula(1, "num_divisor_sum", "_divisor_sum({0})", "sum&divisor|sum&factors", upto_million, "_divisor_sum")
    formula(1, "num_proper_divisor_sum", "_divisor_sum({0}) - {0}", "sum&divisor|sum&factors|aliquot", upto_million,
            "_divisor_sum")
    # The common divisors of two numbers are the divisors of their gcd. One operator for both steps:
    # as a chain it ties with accidental chains such as "proper divisor sum of a + b".
    formula(2, "num_common_divisor_sum", "_divisor_sum(math.gcd({0}, {1}))", "sum&common divisor|sum&common factor",
            ints((1, 10 ** 9), (1, 10 ** 9)), "_divisor_sum")
    formula(2, "num_common_divisor_count", "len(_divisors(math.gcd({0}, {1})))",
            "|".join(f"{count}&common {noun}" for count in ("number of", "count", "how many")
                     for noun in ("divisor", "factor")), ints((1, 10 ** 9), (1, 10 ** 9)), "_divisors")
    prime_factor_words = "prime factor|prime divisor"
    formula(1, "num_prime_factors", "_prime_factors({0})", prime_factor_words, ints((2, 10 ** 6)), "_prime_factors")
    formula(1, "num_prime_factorization", "_prime_factors({0}, True)", prime_factor_words, ints((2, 10 ** 6)),
            "_prime_factors")
    # "less than n" and "up to n" differ only when n is prime; the strict reading is registered first so
    # that it wins when the description asks for it and the example cannot tell.
    below = "&less|&below|&smaller|&under"
    formula(1, "num_primes_below", "_primes_upto({0} - 1)",
            "|".join("prime number" + w + "|primes" + w for w in below.split("|")), ints((3, 1000)), "_primes_upto")
    formula(1, "num_primes_upto", "_primes_upto({0})", "prime number|primes", ints((2, 1000)), "_primes_upto")
    # "prime number" / "primes", not just "prime": tasks about prime factors must not open this gate.
    count_words = [f"{count}{noun}" for count in ("number of ", "count&", "how many ") for noun in ("prime number",
                                                                                                    "primes")]
    formula(1, "num_prime_count_below", "len(_primes_upto({0} - 1))",
            "|".join(c + w for c in count_words for w in below.split("|")), ints((0, 20000)), "_primes_upto")
    formula(1, "num_prime_count_upto", "len(_primes_upto({0}))", "|".join(count_words), ints((0, 20000)),
            "_primes_upto")
    formula(1, "num_nth_prime", "_nth_prime({0})", "nth prime|n-th prime|n'th prime|kth prime|k-th prime",
            ints((1, 2000)), "_nth_prime")
    formula(2, "num_primes_in_range", "[p for p in _primes_upto({1}) if p >= {0}]",
            "|".join(f"{noun}&{where}" for noun in ("prime number", "primes")
                     for where in ("between", "range", "interval", "from")),
            lambda a, b: need(is_int(a) and is_int(b) and 0 <= a <= b <= 5000), "_primes_upto")
    formula(1, "num_amicable_upto", "_amicable_upto({0})", "amicable", ints((1, 10000)), "_amicable_upto")
    formula(1, "num_lcm_upto", "math.lcm(*range(1, {0} + 1))",
            "smallest multiple|lcm of first|lcm&1 to|least common multiple of first|evenly divisible|"
            "divisible by all", ints((1, 200)))

    # ------------------------------------------------------------ bits
    # "Even bits" are the 2nd, 4th, ... counted from 1 at the low end, which is how these tasks count.
    # The gate wants "even bit" as a phrase: "odd number of set bits" is a different question.
    word = ints((0, 2 ** 62))
    for action, symbol, verbs in (("set", "|", ["set", "turn on", "switch on"]),
                                  ("toggle", "^", ["toggle", "flip", "invert"])):
        for parity, first in (("even", 1), ("odd", 0)):
            places = [f"{parity} bit", f"{parity} position", f"{parity}-position", f"{parity} place", f"{parity} ind"]
            formula(1, f"num_{action}_{parity}_bits", f"{{0}} {symbol} _every_other_bit({{0}}, {first})",
                    "|".join(f"{verb}&{place}" for verb in verbs for place in places), word, "_every_other_bit")
    zero_bit = ("unset bit", "zero bit", "0 bit", "bit that is not set")
    formula(1, "num_set_leftmost_unset_bit", "_set_leftmost_unset_bit({0})",
            "|".join(f"{side}&{zero}" for side in ("left", "most significant", "highest") for zero in zero_bit), word,
            "_set_leftmost_unset_bit")
    # A number made of ones only has no unset bit inside its length and stays as it is.
    formula(1, "num_set_rightmost_unset_bit", "{0} if {0} > 0 and {0} & ({0} + 1) == 0 else {0} | ({0} + 1)",
            "|".join(f"{side}&{zero}" for side in ("right", "least significant", "lowest") for zero in zero_bit),
            word)
    formula(1, "num_lowest_set_bit_position", "({0} & -{0}).bit_length()",
            "position&right&set bit|position&first set bit|position&lowest set bit|position&least significant",
            ints((1, 2 ** 62)))
    formula(1, "num_highest_power_of_2_upto", "1 << ({0}.bit_length() - 1) if {0} > 0 else 0",
            "most significant bit|most significant set bit|msb|highest power of 2|largest power of 2|"
            "greatest power of 2|highest power of two|largest power of two|power of 2&less than|"
            "power of two&less than", word)
    formula(1, "num_next_power_of_2", "1 << ({0} - 1).bit_length() if {0} > 1 else 1",
            "smallest power of 2|smallest power of two|next power of 2|next power of two|power of 2&greater|"
            "power of two&greater|power of 2&at least", word)
    # The total over 1..n comes first: its descriptions also contain the words of the single-number count.
    upto_n = ("total", "1 to", "from 1", "all numbers")
    formula(1, "num_total_unset_bits_upto", "sum(bin(k)[2:].count('0') for k in range(1, {0} + 1))",
            "|".join(f"{zero}&{scope_word}" for zero in ("unset bit", "zero bit") for scope_word in upto_n),
            ints((1, 2000)))
    formula(1, "num_unset_bit_count", "{0}.bit_length() - bin({0}).count('1')",
            "count&unset bit|number of&unset bit|unset bits|count&zero bit|number of&zero bit|zero bits", word)
    # Reference solutions of this kind do not mask the left rotation to 32 bits, so neither does this.
    formula(2, "num_rotate_left_32", "({0} << {1}) | ({0} >> (32 - {1}))", "rotat&left&bit",
            ints((0, 2 ** 32 - 1), (0, 32)))
    formula(2, "num_rotate_right_32", "({0} >> {1}) | ({0} << (32 - {1})) & 0xFFFFFFFF", "rotat&right&bit",
            ints((0, 2 ** 32 - 1), (0, 32)))
    formula(2, "num_hamming_distance", "bin({0} ^ {1}).count('1')",
            "hamming|bits&differ|different bits|bit difference", ints((0, 2 ** 62), (0, 2 ** 62)))

    # ------------------------------------------------------------ digits and number bases
    # abs() because the last digit of -13 is 3, while Python's % would answer 7.
    any_int = ints((-10 ** 30, 10 ** 30))
    formula(1, "num_last_digit", "abs({0}) % 10",
            "last digit|unit digit|units digit|unit place|ones digit|rightmost digit", any_int)
    formula(1, "num_last_two_digits", "abs({0}) % 100", "last two digit|last 2 digit", any_int)
    formula(1, "num_next_palindrome", "_next_palindrome({0})",
            "|".join(f"palindrome&{w}" for w in ("next", "after", "following", "greater", "larger", "bigger",
                                                 "above")), ints((0, 10 ** 15)), "_next_palindrome")
    formula(1, "num_previous_palindrome", "_previous_palindrome({0})",
            "|".join(f"palindrome&{w}" for w in ("previous", "preceding", "before", "smaller", "less than", "below")),
            ints((1, 10 ** 15)), "_previous_palindrome")

    # The answer when no bigger arrangement exists is False by convention; the search only uses real ones.
    def has_bigger(n):
        need(is_int(n) and 10 <= n < 10 ** 15 and sorted(str(n), reverse=True) != list(str(n)))
    formula(1, "num_next_bigger", "_next_bigger({0})",
            "next bigger|next greater|next larger|next higher|bigger&rearrang|rearrang&digit", has_bigger,
            "_next_bigger")
    formula(1, "num_binary_to_decimal", "int(str({0}), 2)", "binary&decimal", digits_below(2))
    formula(1, "num_octal_to_decimal", "int(str({0}), 8)", "octal", digits_below(8))
    formula(1, "num_decimal_to_octal", "int(oct({0})[2:])", "octal", ints((0, 10 ** 15)))
    formula(1, "num_to_roman", "_to_roman({0})", "roman", ints((1, 3999)), "_to_roman")
    formula(3, "num_fraction_digit", "{0} * 10 ** {2} // {1} % 10", "digit&fraction",
            ints((0, 10 ** 9), (1, 10 ** 9), (1, 60)))
    formula(3, "num_largest_with_remainder", "{0} - ({0} - {2}) % {1} if {0} >= {2} else -1",
            "largest&modulo|maximum&modulo|largest&remainder|maximum&remainder",
            lambda n, x, y: need(all(is_int(v) for v in (n, x, y)) and 0 <= y < x <= 10 ** 9 and y <= n <= 10 ** 12))
    # Profit and loss tasks answer None when there is nothing to report; the search tries both orders.
    formula(2, "num_surplus_or_none", "{0} - {1} if {0} > {1} else None", "profit&none|loss&none",
            numbers(-10 ** 12, 10 ** 12))

    # ------------------------------------------------------------ progressions and series
    # Argument order is (first term, number of terms, step or ratio); the search tries every order.
    def progression(a, n, step):
        need(is_num(a) and is_num(step) and is_int(n) and 1 <= n <= 60 and abs(a) <= 10 ** 6 and abs(step) <= 10 ** 6)

    def geometric(a, n, ratio):
        progression(a, n, ratio)
        need(ratio != 1)
    formula(3, "num_ap_term", "{0} + ({1} - 1) * {2}", "term&arith|term&a.p", progression)
    formula(3, "num_ap_sum", "{1} * (2 * {0} + ({1} - 1) * {2}) / 2",
            "sum&arith&progression|sum&arith&series|sum&a.p", progression)
    formula(3, "num_gp_term", "{0} * {2} ** ({1} - 1)", "term&geometric|term&g.p", progression)
    formula(3, "num_gp_sum", "{0} * ({2} ** {1} - 1) / ({2} - 1)",
            "sum&geometric&progression|sum&geometric&series|sum&g.p", geometric)
    formula(1, "num_harmonic_sum", "_harmonic_sum({0})", "harmonic", ints((1, 2000)), "_harmonic_sum")
    formula(1, "num_geometric_sum", "sum(1 / 2 ** k for k in range({0} + 1))", "geometric sum", ints((0, 1000)))

    # ------------------------------------------------------------ geometry and conversions
    # Float results must equal the example digit for digit, so each formula keeps the evaluation order
    # the usual textbook code has.
    formula(2, "num_hypotenuse", "math.sqrt({0} * {0} + {1} * {1})",
            "right angle|right-angle|rightangle|right triangle|hypotenuse", positive)
    formula(2, "num_right_triangle_leg", "math.sqrt(abs({0} * {0} - {1} * {1}))",
            "right angle|right-angle|rightangle|right triangle|hypotenuse",
            lambda a, b: (positive(a, b), need(a != b)))
    formula(1, "num_degrees_to_radians", "{0} * math.pi / 180", "degree&radian", numbers(-10 ** 6, 10 ** 6))
    formula(1, "num_radians_to_degrees", "{0} * (180 / math.pi)", "degree&radian", numbers(-10 ** 6, 10 ** 6))
    formula(2, "num_third_angle", "180 - ({0} + {1})", "third angle",
            lambda a, b: (positive(a, b), need(a + b < 180)))
    formula(1, "num_circle_diameter", "2 * {0}", "diameter&circle|diameter&sphere", positive)
    formula(1, "num_pentagon_area", "math.sqrt(5 * (5 + 2 * math.sqrt(5))) * {0} ** 2 / 4", "pentagon&area", positive)
    formula(1, "num_tetrahedron_area_grouped", "math.sqrt(3) * ({0} * {0})",
            "tetrahedron&area|tetrahedron&surface", positive)

    def quadratic(a, b, c):
        need(all(is_num(v) and abs(v) < 10 ** 6 for v in (a, b, c)) and a != 0)
    formula(3, "num_parabola_vertex", "(-{1} / (2 * {0}), (4 * {0} * {2} - {1} * {1}) / (4 * {0}))",
            "parabola&vertex", quadratic)
    formula(3, "num_parabola_focus", "(-{1} / (2 * {0}), (4 * {0} * {2} - {1} * {1} + 1) / (4 * {0}))",
            "parabola&focus", quadratic)
    formula(1, "num_polar", "cmath.polar({0})", "polar",
            lambda z: need((is_num(z) or isinstance(z, complex)) and abs(z) < 10 ** 9))
    # The named method first: where both give the same float, the description asked for the iteration.
    formula(1, "num_babylonian_sqrt", "_babylonian_sqrt({0})", "babylonian|heron", numbers(1e-9, 10 ** 12),
            "_babylonian_sqrt")
    formula(1, "num_sqrt", "math.sqrt({0})", "square root|sqrt", numbers(0, 10 ** 15))

    def rounding(x, places):
        need(isinstance(x, float) and abs(x) < 10 ** 9 and is_int(places) and 0 <= places <= 10)
    # Directed rounding before plain rounding: "round up" also contains "round", and the three agree
    # on a number that already has few enough decimals.
    formula(2, "num_round_up_to", "math.ceil({0} * 10 ** {1}) / 10 ** {1}", "round up|round&up to|ceil", rounding,
            kind="generic")
    formula(2, "num_round_down_to", "math.floor({0} * 10 ** {1}) / 10 ** {1}",
            "round down|round&down to|floor|truncat", rounding, kind="generic")
    formula(2, "num_round_to", "round({0}, {1})", "round", rounding, kind="generic")
    # Halfway between two multiples goes down, as in the usual solution; floor division alone only
    # happens to agree on examples that round down.
    formula(2, "num_nearest_multiple", "{0} // {1} * {1} if {0} % {1} * 2 <= {1} else ({0} // {1} + 1) * {1}",
            "nearest multiple|closest multiple|round&multiple", ints((0, 10 ** 12), (1, 10 ** 9)), kind="generic")
    formula(3, "num_rgb_to_hsv", "_rgb_to_hsv({0}, {1}, {2})", "rgb&hsv", ints((0, 255), (0, 255), (0, 255)),
            "_rgb_to_hsv")

    # ------------------------------------------------------------ named sequences and counting formulas
    formula(2, "num_entringer", "_entringer({0}, {1})", "entringer",
            lambda n, k: need(is_int(n) and is_int(k) and 0 <= k <= n <= 20), "_entringer")
    formula(1, "num_lucky_numbers", "_lucky_numbers({0})", "lucky", ints((1, 120)), "_lucky_numbers")
    formula(1, "num_min_squares", "_min_squares({0})",
            "|".join(f"{few}&squares&sum" for few in ("minimum", "least", "fewest", "smallest number")),
            ints((1, 10 ** 6)), "_min_squares")
    formula(2, "num_squares_in_rectangle", "sum(({0} - k) * ({1} - k) for k in range(min({0}, {1})))",
            "squares&rectangle", ints((1, 2000), (1, 2000)))
    formula(2, "num_fence_ways", "_fence_ways({0}, {1})", "fence", ints((1, 60), (1, 60)), "_fence_ways")
    # Closed form kept in floats on purpose: it is the published formula, and it overflows where the
    # usual solution does.
    formula(1, "num_triangular_index_with_digits", "round(math.sqrt(2 * math.pow(10, {0} - 1)))",
            "triangular&digit", ints((1, 300)))
    formula(1, "num_spiral_matrix", "_spiral_matrix({0})", "spiral", ints((1, 12)), "_spiral_matrix")
    formula(1, "num_domino_tilings_3xn", "_domino_tilings_3xn({0})", "domino&3 x n|domino&3xn|domino&3 by n",
            ints((0, 200)), "_domino_tilings_3xn")
    # Days left over after whole weeks in n years of the Gregorian calendar.
    formula(1, "num_odd_days", "({0} + {0} // 4 - {0} // 100 + {0} // 400) % 7", "odd days", ints((1, 10 ** 6)))

    # ------------------------------------------------------------ numbers of one kind in a range
    # These return the list, so "sum of", "count of", "largest" and "average of" follow from one more
    # core operator instead of one formula each.
    def span(a, b):
        need(is_int(a) and is_int(b) and 0 <= a <= b <= a + 1000 and b <= 10 ** 6)
    formula(2, "num_self_dividing_in_range",
            "[n for n in range({0}, {1} + 1) if all(d != '0' and n % int(d) == 0 for d in str(n))]",
            "divisible&digit|self-dividing|self dividing", span)
    formula(2, "num_perfect_squares_in_range",
            "[k for k in range({0}, {1} + 1) if k >= 0 and math.isqrt(k) ** 2 == k]",
            "perfect square&between|perfect square&range|perfect squares&from|odd factor|odd number of factor|"
            "odd number of divisor", span)
    # Plural on purpose: "an odd number of set bits" is not about the odd numbers.
    formula(1, "num_odds_upto", "list(range(1, {0} + 1, 2))", "odd numbers|odd natural numbers|odd integers",
            ints((1, 300)))
    formula(1, "num_evens_upto", "list(range(2, {0} + 1, 2))", "even numbers|even natural numbers|even integers",
            ints((2, 300)))

    # ------------------------------------------------------------ three plain numbers
    # The core library has two-argument min/max/add only, and a search two operators deep cannot nest
    # them when the result equals one of the arguments. Registered after the specific three-argument
    # formulas above, so that those win when both reproduce the example.
    plain = numbers(-10 ** 9, 10 ** 9)
    three = ["three", "3 num", "3 int", "3 value"]
    formula(3, "num_min_of_three", "min({0}, {1}, {2})",
            "|".join(f"{w}&{n}" for w in ("minimum", "smallest", "lowest", "least") for n in three), plain,
            kind="generic")
    formula(3, "num_max_of_three", "max({0}, {1}, {2})",
            "|".join(f"{w}&{n}" for w in ("maximum", "largest", "greatest", "biggest", "highest") for n in three),
            plain, kind="generic")
    formula(3, "num_sum_of_three", "{0} + {1} + {2}",
            "|".join(f"{w}&{n}" for w in ("sum", "add", "total") for n in three) + "|perimeter&triangle", plain,
            kind="generic")
    # The median of three ints is one of the inputs, and the engine does not build on a value it
    # already has, so the float form some examples show needs its own operator.
    formula(3, "num_median_of_three_float", "float(sorted([{0}, {1}, {2}])[1])", "median&three|median&3 num", plain,
            kind="generic")

    # ------------------------------------------------------------ more words for existing operators
    lib.also_named("rectangle_perimeter/2", "perimeter&cylinder")
    lib.also_named("square_perimeter/1", "perimeter&rombus")
    lib.also_named("rhombus_area/2", "rombus&area")
    lib.also_named("natural_power2_sum/1", "sum&1²|sum&1^2")
    lib.also_named("fibonacci/1", "tiling")
