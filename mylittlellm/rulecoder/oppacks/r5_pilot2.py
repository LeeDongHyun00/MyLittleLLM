"""Round 5 pilot 2 operators: textbook formulas and conventions, results rounded to a stated precision, named
number and string definitions, and the conventional or robust variants of list, string and dictionary operations.

Every operator is gated: the description has to name the concept before the search tries it. The search function
of each operator is compiled from the template it emits, so the program verified on the example is by construction
the program that is shipped; the guard only bounds where the search may try it and how much work a call can do.
"""
import importlib

# Modules the engine imports automatically in generated answers; templates and helpers may use them freely.
MODULES = ("bisect", "collections", "datetime", "functools", "itertools", "math", "operator", "re", "statistics",
           "string")

# Phrases that state the precision of a result. The engine ranks f(x) above round(f(x), k) whenever both reproduce
# the example, so a whole-valued example such as 77.0 ships the unrounded formula. A formula that is also registered
# rounded, named by its own words plus the precision phrase, wins that tie on the more specific wording.
PLACES = {
    1: "one decimal|1 decimal|single decimal|nearest tenth|one digit after|1 digit after|one place after|1 d.p"
       "|to one place|to 1 place|precision of 1",
    2: "two decimal|2 decimal|nearest hundredth|two digits after|2 digits after|two places after|2 d.p|nearest cent"
       "|to two places|to 2 places|precision of 2",
    3: "three decimal|3 decimal|nearest thousandth|three digits after|3 digits after|three places after|3 d.p"
       "|to three places|to 3 places|precision of 3",
}

# Existing float formulas that tasks commonly ask for at a stated precision.
ROUNDED_EXISTING = (
    "circle_area/1", "circumference/1", "sphere_surface/1", "sphere_volume/1", "tetrahedron_area/1",
    "tetrahedron_volume/1", "mean/1", "cylinder_lateral/2", "cylinder_surface/2", "cylinder_volume/2",
    "cone_lateral/2", "cone_surface/2", "cone_volume/2", "pyramid_volume/2", "ellipse_area/2", "sector_area/2",
    "arc_length/2", "polygon_area/2", "triangle_area/2", "trapezium_area/3", "num_hypotenuse/2",
    "num_right_triangle_leg/2", "num_degrees_to_radians/1", "num_radians_to_degrees/1", "num_pentagon_area/1",
    "num_sqrt/1", "num_harmonic_sum/1", "num_ap_sum/3", "num_gp_sum/3", "seq_stdev_sample/1")

# ---------------------------------------------------------------- helper sources
# Pasted into generated answers, so they rely on builtins and the modules above only, never modify an argument,
# and stay linear or logarithmic where hidden tests may call them with much larger values than the example.
HERON = '''def _r52_heron(a, b, c):
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))
'''
COLLATZ_STEPS = '''def _r52_collatz_steps(n):
    steps = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps
'''
COLLATZ_SEQUENCE = '''def _r52_collatz_sequence(n):
    terms = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        terms.append(n)
    return terms
'''
# Climbing n stairs in steps of 1 or 2: ways(0) = ways(1) = 1, then the Fibonacci recurrence.
STAIRS = '''def _r52_stair_ways(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a
'''
NEXT_PRIME = '''def _r52_next_prime(n):
    k = max(n + 1, 2)
    while any(k % d == 0 for d in range(2, math.isqrt(k) + 1)):
        k += 1
    return k
'''
# Legendre's formula: every multiple of 5, 25, 125, ... below n contributes one factor 5 per power.
FACTORIAL_ZEROS = '''def _r52_factorial_zeros(n):
    count, power = 0, 5
    while power <= n:
        count += n // power
        power *= 5
    return count
'''
# Counted per digit length, so 10 ** 12 costs a dozen steps instead of a loop over every number.
DIGITS_UPTO = '''def _r52_digits_upto(n):
    total, width, start = 0, 1, 1
    while start <= n:
        total += (min(n, start * 10 - 1) - start + 1) * width
        width += 1
        start *= 10
    return total
'''
MINUTES = '''def _r52_minutes(text):
    hours, minutes = text.strip().split(':')[:2]
    return int(hours) * 60 + int(minutes)
'''
TWELVE_HOUR = '''def _r52_to_12_hour(text, pad=False):
    hours, rest = text.strip().split(':', 1)
    hour = int(hours)
    return ('%02d' if pad else '%d') % (hour % 12 or 12) + ':' + rest + (' AM' if hour < 12 else ' PM')
'''
TWENTY_FOUR_HOUR = '''def _r52_to_24_hour(text):
    match = re.fullmatch(r'\\s*(\\d{1,2}):(\\d{2})(:\\d{2})?\\s*([AaPp])\\.?[Mm]\\.?\\s*', text)
    hour = int(match.group(1)) % 12 + (12 if match.group(4) in 'Pp' else 0)
    return '%02d:%s%s' % (hour, match.group(2), match.group(3) or '')
'''
COLUMN_LABEL = '''def _r52_column_label(n):
    label = ''
    while n > 0:
        n, rest = divmod(n - 1, 26)
        label = chr(65 + rest) + label
    return label
'''
COLUMN_NUMBER = '''def _r52_column_number(label):
    n = 0
    for c in label:
        n = n * 26 + ord(c.upper()) - 64
    return n
'''
CAESAR = '''def _r52_caesar(text, shift):
    out = []
    for c in text:
        if 'a' <= c <= 'z':
            out.append(chr((ord(c) - 97 + shift) % 26 + 97))
        elif 'A' <= c <= 'Z':
            out.append(chr((ord(c) - 65 + shift) % 26 + 65))
        else:
            out.append(c)
    return ''.join(out)
'''
# Each letter is mirrored inside its own case: ord('a') + ord('z') == 219 and ord('A') + ord('Z') == 155.
ATBASH = '''def _r52_atbash(text):
    out = []
    for c in text:
        if 'a' <= c <= 'z':
            out.append(chr(219 - ord(c)))
        elif 'A' <= c <= 'Z':
            out.append(chr(155 - ord(c)))
        else:
            out.append(c)
    return ''.join(out)
'''
PIG_LATIN = '''def _r52_pig_latin(word):
    for i, c in enumerate(word):
        if c.lower() in 'aeiou':
            return word + 'way' if i == 0 else word[i:] + word[:i] + 'ay'
    return word + 'ay'
'''
# Unequal lengths are the point: the core interleave stops at the shorter sequence and takes lists only.
INTERLEAVE = '''def _r52_interleave(a, b):
    out = []
    for i in range(max(len(a), len(b))):
        out.extend(a[i:i + 1])
        out.extend(b[i:i + 1])
    return ''.join(out) if isinstance(a, str) else type(a)(out)
'''
# Python's str.center puts the odd fill character on the left for odd widths; this puts it on the right always.
CENTER = '''def _r52_center(text, width, fill=' '):
    pad = max(0, width - len(text))
    return fill * (pad // 2) + text + fill * (pad - pad // 2)
'''
# A new record only has to beat the last record, and a leader the last leader found from the right.
PREFIX_MAXIMA = '''def _r52_prefix_maxima(items):
    records = []
    for value in items:
        if not records or value > records[-1]:
            records.append(value)
    return records
'''
LEADERS = '''def _r52_leaders(items):
    leaders = []
    for value in reversed(items):
        if not leaders or value > leaders[-1]:
            leaders.append(value)
    return leaders[::-1]
'''
FIRST_REPEAT = '''def _r52_first_repeat(items):
    seen = set()
    for item in items:
        if item in seen:
            return item
        seen.add(item)
    return None
'''
KADANE = '''def _r52_max_subarray(items):
    best = current = items[0]
    for value in items[1:]:
        current = max(value, current + value)
        best = max(best, current)
    return best
'''
LONGEST_RISE = '''def _r52_longest_rise(items):
    best = run = 0
    for i, value in enumerate(items):
        run = run + 1 if i and items[i - 1] < value else 1
        best = max(best, run)
    return best
'''
CONSECUTIVE_RUNS = '''def _r52_consecutive_runs(items):
    runs = []
    for value in items:
        if runs and value == runs[-1][-1] + 1:
            runs[-1].append(value)
        else:
            runs.append([value])
    return runs
'''
IS_IPV4 = '''def _r52_is_ipv4(text):
    parts = text.split('.')
    return len(parts) == 4 and all(
        p.isdigit() and p.isascii() and (p == '0' or not p.startswith('0')) and int(p) <= 255 for p in parts)
'''
MOUNTAIN = '''def _r52_is_mountain(items):
    i, n = 0, len(items)
    while i + 1 < n and items[i] < items[i + 1]:
        i += 1
    if i == 0 or i == n - 1:
        return False
    while i + 1 < n and items[i] > items[i + 1]:
        i += 1
    return i == n - 1
'''
# Integer bisection, so the answer is exact for numbers far beyond float precision.
IS_CUBE = '''def _r52_is_cube(n):
    m = abs(n)
    low, high = 0, 1
    while high ** 3 < m:
        high *= 2
    while low < high:
        middle = (low + high) // 2
        if middle ** 3 < m:
            low = middle + 1
        else:
            high = middle
    return low ** 3 == m
'''
IS_UGLY = '''def _r52_is_ugly(n):
    if n < 1:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1
'''
POWER_OF = '''def _r52_is_power_of(n, base):
    if n < 1:
        return False
    while n % base == 0:
        n //= base
    return n == 1
'''
# A set of the values seen so far makes the pair test linear.
PAIR_SUM = '''def _r52_has_pair_sum(items, target):
    seen = set()
    for value in items:
        if target - value in seen:
            return True
        seen.add(value)
    return False
'''
SPIRAL = '''def _r52_spiral_order(matrix):
    rows = [list(row) for row in matrix]
    out = []
    while rows:
        out.extend(rows.pop(0))
        rows = [list(column) for column in zip(*rows)][::-1]
    return out
'''
BORDER = '''def _r52_border(matrix):
    last_row, last_column = len(matrix) - 1, len(matrix[0]) - 1
    return [v for r, row in enumerate(matrix) for c, v in enumerate(row)
            if r in (0, last_row) or c in (0, last_column)]
'''
# Top row, right column, bottom row backwards, left column upwards; a single row or column is listed once.
BORDER_CLOCKWISE = '''def _r52_border_clockwise(matrix):
    rows, columns = len(matrix), len(matrix[0])
    if rows == 1 or columns == 1:
        return [v for row in matrix for v in row]
    return (list(matrix[0]) + [matrix[r][-1] for r in range(1, rows - 1)] + list(matrix[-1][::-1])
            + [matrix[r][0] for r in range(rows - 2, 0, -1)])
'''
GROUP_BY = '''def _r52_group_by(items, key):
    groups = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    return groups
'''
SUM_BY_KEY = '''def _r52_sum_by_key(pairs):
    totals = {}
    for key, amount in pairs:
        totals[key] = totals.get(key, 0) + amount
    return totals
'''
GET_PATH = '''def _r52_get_path(data, keys):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data
'''
COUNT_LEAVES = '''def _r52_count_leaves(data):
    return sum(_r52_count_leaves(v) if isinstance(v, dict) else 1 for v in data.values())
'''
NESTED_DICT_SUM = '''def _r52_nested_dict_sum(data):
    return sum(_r52_nested_dict_sum(v) if isinstance(v, dict) else v for v in data.values())
'''


def cross(*groups):
    """Keywords that need one alternative of each group: cross('a|b', 'x') is 'a&x|b&x'."""
    alternatives = [""]
    for group in groups:
        alternatives = [f"{a}&{b}" if a else b for a in alternatives for b in group.split("|")]
    return "|".join(alternatives)


def register(lib):
    need, is_num, is_int, is_seq, seqseq = lib.need, lib.is_num, lib.is_int, lib.is_seq, lib.seqseq
    scope = {name: importlib.import_module(name) for name in MODULES}
    for source in (HERON, COLLATZ_STEPS, COLLATZ_SEQUENCE, STAIRS, NEXT_PRIME, FACTORIAL_ZEROS, DIGITS_UPTO, MINUTES,
                   TWELVE_HOUR, TWENTY_FOUR_HOUR, COLUMN_LABEL, COLUMN_NUMBER, CAESAR, ATBASH, PIG_LATIN, INTERLEAVE,
                   CENTER, PREFIX_MAXIMA, LEADERS, FIRST_REPEAT, KADANE, LONGEST_RISE, CONSECUTIVE_RUNS, IS_IPV4,
                   MOUNTAIN, IS_CUBE, IS_UGLY, POWER_OF, PAIR_SUM, SPIRAL, BORDER, BORDER_CLOCKWISE, GROUP_BY,
                   SUM_BY_KEY, GET_PATH, COUNT_LEAVES, NESTED_DICT_SUM):
        exec(source, scope)
    makers = {1: lib.unary, 2: lib.binary, 3: lib.ternary}

    def publish(arity, name, fn, template, keywords, kind, helper):
        options = {"gated": True, "kind": kind}
        if helper:
            options["helper"] = helper
        makers[arity]("r52_" + name, fn, template, keywords, **options)

    def rounded(arity, name, fn, template, keywords, kind, helper):
        for places, phrases in PLACES.items():
            def run(*args, places=places):
                value = fn(*args)
                need(isinstance(value, float))
                return round(value, places)
            publish(arity, f"{name}_r{places}", run, f"round({template}, {places})", cross(keywords, phrases), kind,
                    helper)

    def add(arity, name, guard, template, keywords, *, helpers=(), kind="knowledge", valid=None, places=False):
        params = ("a0", "a1", "a2")[:arity]
        body = eval(f"lambda {', '.join(params)}: {template.format(*params)}", scope)

        def fn(*args):
            need(guard(*args))
            value = body(*args)
            need(valid is None or valid(value))
            return value
        helper = "".join(helpers) or None
        publish(arity, name, fn, template, keywords, kind, helper)
        if places:
            rounded(arity, name, fn, template, keywords, kind, helper)

    def predicate(arity, name, guard, template, keywords, helpers=()):
        # A 0/1 or None result would never equal a True/False example, so only real booleans count.
        add(arity, name, guard, template, keywords, helpers=helpers, valid=lambda v: v is True or v is False)

    def also(key, keywords):
        # The pack also loads without the round-3 packs; a missing rule then simply has nothing to extend.
        try:
            lib.also_named(key, keywords)
        except KeyError:
            pass

    # ------------------------------------------------------------ guards
    def text(x, low=1, high=600):
        return isinstance(x, str) and low <= len(x) <= high

    def seq(x, low=1, high=300):
        return is_seq(x) and low <= len(x) <= high

    def nums(x, low=1, high=300):
        return seq(x, low, high) and all(is_num(v) for v in x)

    def ints(x, low=1, high=300):
        return seq(x, low, high) and all(is_int(v) for v in x)

    def strs(x, low=1, high=300):
        return seq(x, low, high) and all(isinstance(v, str) for v in x)

    def number(v, low=-10 ** 9, high=10 ** 9):
        return is_num(v) and low <= v <= high

    def whole(v, low=0, high=10 ** 9):
        return is_int(v) and low <= v <= high

    def scalar(v):
        return is_num(v) or isinstance(v, str)

    def flat(x, low=1, high=300):
        return seq(x, low, high) and all(scalar(v) for v in x)

    def matrix(x, cell=is_num, size=30):
        return (seqseq(x) and len(x) <= size and all(1 <= len(r) <= size for r in x) and len({len(r) for r in x}) == 1
                and all(cell(v) for r in x for v in r))

    def hhmm(t):
        return text(t, 3, 8) and scope["re"].fullmatch(r"\s*\d{1,2}:\d{2}(:\d{2})?\s*", t) is not None

    # ------------------------------------------------------------ numbers: unit conversions and money
    temperature = number
    for name, template, words in [
            ("celsius_to_fahrenheit", "{0} * 9 / 5 + 32", cross("celsius|centigrade", "fahrenheit")),
            ("fahrenheit_to_celsius", "({0} - 32) * 5 / 9", cross("celsius|centigrade", "fahrenheit")),
            ("celsius_to_kelvin", "{0} + 273.15", cross("celsius|centigrade", "kelvin")),
            ("kelvin_to_celsius", "{0} - 273.15", cross("celsius|centigrade", "kelvin")),
            ("km_to_miles", "{0} * 0.621371", cross("kilomet|km", "mile")),
            ("miles_to_km", "{0} * 1.609344", cross("kilomet|km", "mile")),
            ("mps_to_kmh", "{0} * 3.6", cross("per second|m/s|mps", "per hour|km/h|kmph|kph|km/hr")),
            ("kmh_to_mps", "{0} / 3.6", cross("per second|m/s|mps", "per hour|km/h|kmph|kph|km/hr"))]:
        add(1, name, temperature, template, words, places=True)

    # Rates are percentages; a rate above 100 is almost always another argument in the wrong position.
    def money(p, r, t):
        return number(p, 0, 10 ** 12) and number(r, 0, 100) and number(t, 0, 1000)
    add(3, "simple_interest", money, "{0} * {1} * {2} / 100", "simple interest", places=True)
    add(3, "compound_amount", money, "{0} * (1 + {1} / 100) ** {2}", "compound", places=True)
    add(3, "compound_interest", money, "{0} * (1 + {1} / 100) ** {2} - {0}", "compound interest", places=True)
    add(2, "bmi", lambda w, h: number(w, 1, 1000) and number(h, 0.3, 3), "{0} / {1} ** 2", "bmi|body mass",
        places=True)
    add(2, "percent_of", lambda a, b: number(a) and number(b) and b != 0, "{0} / {1} * 100", "percent", places=True)
    add(2, "percent_value", lambda p, x: number(p) and number(x), "{0} * {1} / 100", "percent of|% of", places=True)

    # ------------------------------------------------------------ numbers: geometry and calendar
    def points(p, q):
        return (is_seq(p) and is_seq(q) and len(p) == len(q) and 1 <= len(p) <= 4 and all(number(v) for v in p)
                and all(number(v) for v in q))
    add(2, "point_distance", points, "math.dist({0}, {1})",
        "euclidean|" + cross("distance", "point|coordinate|vector"), places=True)
    add(2, "manhattan_distance", points, "sum(abs(a - b) for a, b in zip({0}, {1}))",
        "manhattan|taxicab|city block|city-block|l1 distance")
    add(2, "midpoint", points, "type({0})((a + b) / 2 for a, b in zip({0}, {1}))",
        "midpoint|mid-point|mid point|middle point")
    add(2, "slope", lambda p, q: points(p, q) and len(p) == 2 and p[0] != q[0],
        "({1}[1] - {0}[1]) / ({1}[0] - {0}[0])", "slope|gradient", places=True)
    add(3, "heron_area", lambda a, b, c: all(number(v, 1e-9, 10 ** 6) for v in (a, b, c)),
        "_r52_heron({0}, {1}, {2})",
        "heron|three sides|three side|sides of a triangle|" + cross("triangle", "area", "side"), helpers=(HERON,),
        places=True)
    polygon = lambda n: whole(n, 3, 10 ** 6)  # noqa: E731
    add(1, "interior_angle_sum", polygon, "({0} - 2) * 180",
        "interior angle|sum of the angles|sum of angles|angle sum")
    add(1, "regular_interior_angle", polygon, "({0} - 2) * 180 / {0}",
        "each interior angle|" + cross("interior angle|each angle|one angle", "regular"), places=True)
    add(1, "exterior_angle", polygon, "360 / {0}", "exterior angle", places=True)
    add(1, "polygon_diagonals", polygon, "{0} * ({0} - 3) // 2",
        "number of diagonals|" + cross("diagonal", "polygon|sides|vertices|n-gon"))
    month_words = "days in&month|number of days&month|how many days&month|days&month|length of&month"
    add(2, "days_in_month", lambda y, m: whole(y, 1, 9999) and whole(m, 1, 12),
        "(29 if {0} % 4 == 0 and ({0} % 100 != 0 or {0} % 400 == 0) else 28) if {1} == 2 "
        "else 30 if {1} in (4, 6, 9, 11) else 31", month_words)
    add(1, "days_in_month_common_year", lambda m: whole(m, 1, 12),
        "(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[{0} - 1]", month_words)
    add(3, "day_of_year", lambda y, m, d: whole(y, 1, 9999) and whole(m, 1, 12) and whole(d, 1, 31),
        "datetime.date({0}, {1}, {2}).timetuple().tm_yday",
        "day of the year|day of year|day number|ordinal day|day-of-year|days since the start of the year")
    add(1, "seconds_to_hms", lambda n: whole(n, 0, 10 ** 9), "({0} // 3600, {0} % 3600 // 60, {0} % 60)",
        cross("hour", "minute", "second"))
    add(1, "divmod_60", lambda n: whole(n, 0, 10 ** 9), "divmod({0}, 60)", cross("minute", "hour|second"))
    add(1, "time_to_minutes", hhmm, "_r52_minutes({0})", cross("minute", "hh:mm|time|clock|midnight"),
        helpers=(MINUTES,))
    add(2, "minutes_between", lambda a, b: hhmm(a) and hhmm(b), "_r52_minutes({1}) - _r52_minutes({0})",
        cross("minute", "between|from&to|difference|elapsed|apart|passed|duration|hh:mm"), helpers=(MINUTES,))

    # ------------------------------------------------------------ numbers: digits, primes, sequences
    add(1, "digital_root", lambda n: whole(n, 0, 10 ** 30), "0 if {0} == 0 else 1 + ({0} - 1) % 9",
        "digital root|repeated digit sum|repeatedly&digit|single digit&sum|single digit&add|single-digit&sum")
    add(1, "digit_count_ignoring_sign", lambda n: is_int(n) and abs(n) <= 10 ** 30, "len(str(abs({0})))",
        cross("count|number of|how many", "digit", "negative|minus|sign|absolute"), kind="generic")
    add(1, "factorial_trailing_zeros", lambda n: whole(n, 0, 10 ** 18), "_r52_factorial_zeros({0})",
        cross("trailing zero|trailing 0|zeros at the end|zeroes at the end|ending zero|zeros in the end",
              "factorial|n!"), helpers=(FACTORIAL_ZEROS,))
    add(1, "digits_written_upto", lambda n: whole(n, 0, 10 ** 15), "_r52_digits_upto({0})",
        cross("digit", "1 to n|from 1 to|from 1 through|integers from 1|numbers from 1|written|page number"),
        helpers=(DIGITS_UPTO,))
    collatz = "collatz|3n + 1|3n+1|3 * n + 1|3*n + 1|hailstone"
    add(1, "collatz_steps", lambda n: whole(n, 1, 10 ** 6), "_r52_collatz_steps({0})", collatz,
        helpers=(COLLATZ_STEPS,))
    add(1, "collatz_sequence", lambda n: whole(n, 1, 10 ** 4), "_r52_collatz_sequence({0})", collatz,
        helpers=(COLLATZ_SEQUENCE,))
    add(1, "stair_ways", lambda n: whole(n, 0, 90), "_r52_stair_ways({0})",
        "stair|climb&step|1 or 2 steps|one or two steps", helpers=(STAIRS,))
    add(1, "next_prime", lambda n: is_int(n) and n <= 10 ** 4, "_r52_next_prime({0})",
        "next prime|following prime|" + cross("smallest prime|first prime|nearest prime",
                                              "greater|larger|bigger|after|exceed"), helpers=(NEXT_PRIME,))
    add(1, "sign", lambda x: number(x, -10 ** 18, 10 ** 18), "({0} > 0) - ({0} < 0)",
        "signum|sgn|sign of|sign function|-1 for a negative|-1 if negative|-1 for negative"
        "|-1 if the number is negative|-1 if it is negative|1 for a positive|1 if positive|1 for positive")
    add(1, "gray_code", lambda n: whole(n, 0, 2 ** 62), "{0} ^ ({0} >> 1)", "gray code|gray")
    bit = lambda n, k: is_int(n) and abs(n) <= 2 ** 62 and whole(k, 0, 62)  # noqa: E731
    add(2, "toggle_bit", bit, "{0} ^ (1 << {1})", cross("toggle|flip|invert|complement", "bit"))
    add(2, "set_bit", bit, "{0} | (1 << {1})",
        "set bit k|set the bit|set the k|set the n|set the i|set bit at|turn on&bit|switch on&bit|"
        + cross("set", "kth bit|k-th bit|nth bit|n-th bit|ith bit|i-th bit|bit&position"))
    add(2, "clear_bit", bit, "{0} & ~(1 << {1})", cross("clear|unset|reset|turn off|switch off", "bit"))
    predicate(2, "bit_is_set", bit, "({0} >> {1}) % 2 == 1",
              cross("bit", "is set|is on|is 1|is one") + "|" + cross("check|test", "bit", "set"))
    add(2, "last_digit_of_power", lambda a, b: whole(a, 0, 10 ** 18) and whole(b, 0, 10 ** 18), "pow({0}, {1}, 10)",
        cross("last digit|unit digit|units digit|last decimal digit|rightmost digit|final digit|ones digit",
              "power|raise|exponent|^"))

    # Sums and counts of these lists follow from one more core operator.
    # Bounds for the search only: a list of several hundred multiples is dropped by the engine anyway.
    def multiples(n, k):
        return whole(n, 0, 10 ** 5) and whole(k, 1, 10 ** 4) and n // k <= 300

    def multiples_of_either(n, a, b):
        return whole(n, 0, 3000) and whole(a, 1, 10 ** 4) and whole(b, 1, 10 ** 4) and a != b
    below = "below|less than|under|smaller than|lower than"
    upto = "up to|not exceed|at most|from 1 to|between 1 and|1 to n|less than or equal|no greater than|including"
    add(2, "multiples_below", multiples, "list(range({1}, {0}, {1}))", cross("multiple|divisible", below))
    add(2, "multiples_upto", multiples, "list(range({1}, {0} + 1, {1}))", cross("multiple|divisible", upto))
    add(3, "multiples_of_either_below", multiples_of_either,
        "[k for k in range(1, {0}) if k % {1} == 0 or k % {2} == 0]", cross("multiple|divisible", below))
    add(3, "multiples_of_either_upto", multiples_of_either,
        "[k for k in range(1, {0} + 1) if k % {1} == 0 or k % {2} == 0]", cross("multiple|divisible", upto))
    # Only named when the description fixes the halfway rule; the round-3 rule rounds halfway cases down.
    add(2, "nearest_multiple_half_up", lambda n, k: whole(n, 0, 10 ** 12) and whole(k, 1, 10 ** 9),
        "({0} + {1} // 2) // {1} * {1}",
        cross("nearest multiple|closest multiple|round&multiple",
              "halfway&up|half&up|ties&up|half up|midway&up|halfway&higher|halfway&larger|halfway&greater"),
        kind="generic")

    # ------------------------------------------------------------ number bases and number formatting
    def digits_of(alphabet, prefix):
        def check(s):
            body = s[2:] if text(s, 3, 66) and s[:2].lower() == prefix else s
            return text(body, 1, 64) and all(c in alphabet for c in body.lower())
        return check
    add(1, "parse_binary", digits_of("01", "0b"), "int({0}, 2)", "binary", kind="generic")
    add(1, "parse_hex", digits_of("0123456789abcdef", "0x"), "int({0}, 16)", "hex", kind="generic")
    add(1, "to_hex", lambda n: whole(n, 0, 2 ** 64), "format({0}, 'x')", "hex", kind="generic")
    add(1, "to_hex_upper", lambda n: whole(n, 0, 2 ** 64), "format({0}, 'X')", "hex", kind="generic")
    add(2, "zero_pad", lambda x, w: (whole(x, 0, 10 ** 30) or (text(x, 1, 60) and x.isdigit())) and whole(w, 1, 64),
        "str({0}).zfill({1})",
        "leading zero|zero-pad|zero pad|zeropad|zfill|pad&zero|padded&zero|zeros&pad|fill&zero", kind="generic")
    add(1, "thousands_separator", lambda n: is_int(n) and abs(n) <= 10 ** 30, "format({0}, ',')",
        "thousand|comma&separat|commas&number|digit group", kind="generic")
    add(1, "ordinal", lambda n: whole(n, 0, 10 ** 12),
        "str({0}) + ('th' if {0} % 100 in (11, 12, 13) else {{1: 'st', 2: 'nd', 3: 'rd'}}.get({0} % 10, 'th'))",
        "ordinal|1st|2nd|3rd|st, nd|suffix&st")
    column = "column&label|column&title|column&letter|column&name|spreadsheet|excel"
    add(1, "column_label", lambda n: whole(n, 1, 10 ** 12), "_r52_column_label({0})", column,
        helpers=(COLUMN_LABEL,))
    add(1, "column_number", lambda s: text(s, 1, 10) and s.isalpha() and s.isascii(), "_r52_column_number({0})",
        column, helpers=(COLUMN_NUMBER,))

    # ------------------------------------------------------------ sequences: positions and searching
    # Rotation by k counts k modulo the length, which is the definition the core rules (0 < k < len) only
    # approximate; registered as knowledge so it wins their tie whenever both reproduce the example.
    def rotation(x, k):
        return (seq(x) or text(x)) and is_int(k) and abs(k) <= 10 ** 6
    add(2, "rotate_left", rotation, "({0}[{1} % len({0}):] + {0}[:{1} % len({0})] if len({0}) else {0}[:])",
        "rotat&left|left rotat|rotat&anticlockwise|rotat&anti-clockwise|rotat&counterclockwise|rotat")
    add(2, "rotate_right", rotation, "({0}[-{1} % len({0}):] + {0}[:-{1} % len({0})] if len({0}) else {0}[:])",
        "rotat&right|right rotat|rotat&clockwise|rotat")

    def member(x, v):
        return (flat(x) and scalar(v)) or (text(x) and text(v, 1, 60))

    def single(x, v):
        return (flat(x) and scalar(v)) or (text(x) and text(v, 1, 1))
    add(2, "index_or_minus1", member, "({0}.index({1}) if {1} in {0} else -1)",
        "binary search|linear search|sequential search|"
        + cross("index|position", "-1|not found|not present|does not occur|doesn't occur|not in the|absent|missing"
                                  "|does not exist|doesn't exist|otherwise|not contain"), kind="generic")
    add(2, "last_index_or_minus1", single, "({0}.rfind({1}) if isinstance({0}, str) else len({0}) - 1 - "
        "{0}[::-1].index({1}) if {1} in {0} else -1)",
        "last occurrence|last index|last position|rightmost|final occurrence|last appearance", kind="generic")
    add(2, "positions_of", single, "[i for i, e in enumerate({0}) if e == {1}]",
        "all indices|all the indices|every index|all positions|all the positions|every position|each index"
        "|each position|indices of|indexes of|positions of|list of indices|list of positions|all index"
        "|indices at which|index at which|"
        + cross("index|position", "every occurrence|each occurrence|all occurrences"), kind="generic")
    add(1, "first_peak_index", nums,
        "next((i for i in range(1, len({0}) - 1) if {0}[i - 1] < {0}[i] > {0}[i + 1]), -1)",
        "peak|local max|" + cross("greater than|larger than|bigger than", "both&neighbo|its neighbo|left and right"))
    add(2, "take_until", lambda x, v: flat(x) and scalar(v), "({0}[:{0}.index({1})] if {1} in {0} else {0}[:])",
        "before the first|until|up to the first|prior to the first|stop value|stops at|takewhile|take while",
        kind="generic")
    add(1, "first_repeat_or_none", lambda x: flat(x), "_r52_first_repeat({0})",
        cross("first", "repeat|duplicate|already appeared|appeared earlier|appeared before|seen before|occurs again"
                       "|appears again|already seen", "none"), helpers=(FIRST_REPEAT,))
    fixed = ("fixed point|equal to its index|equal to its own index|equals its index|equal to i|equal to the index"
             "|equal to their index|same as its index|equal to its position|equals its position|a[i] == i|a[i] = i")
    add(1, "first_unique_char_or_empty", text,
        "(lambda counts: next((c for c in {0} if counts[c] == 1), ''))(collections.Counter({0}))",
        cross("non-repeat|non repeat|not repeat|nonrepeat|unique|only once|occurs once|appears once|exactly once"
              "|does not repeat|doesn't repeat|never repeat", "empty string|''|\"\""))
    add(1, "count_fixed_points", ints, "sum(v == i for i, v in enumerate({0}))", fixed)
    add(1, "fixed_points", ints, "[i for i, v in enumerate({0}) if v == i]",
        fixed)

    # ------------------------------------------------------------ sequences: scans and aggregates
    add(1, "longest_run", lambda x: seq(x, 1, 600) or text(x),
        "max((len(list(g)) for _, g in itertools.groupby({0})), default=0)",
        "longest run|longest streak|max run|maximum run|run of equal|run of identical|run of the same|streak of|"
        + cross("longest", "consecutive&same|consecutive&equal|consecutive&identical|consecutive&repeat"
                           "|adjacent&equal|adjacent&same|adjacent&identical|equal adjacent"))
    add(1, "longest_increasing_run", nums, "_r52_longest_rise({0})",
        cross("longest|maximum length|max length", "increasing|rising|ascending",
              "contiguous|consecutive|stretch|run|subarray|sub-array|continuous|streak|window|segment") + "|"
        + cross("longest", "contiguous|stretch|run|subarray|sub-array|streak|segment|continuous",
                "greater than the one before|greater than the previous|larger than the previous"
                "|bigger than the previous|greater than its predecessor"), helpers=(LONGEST_RISE,))
    add(1, "max_subarray_sum", nums, "_r52_max_subarray({0})",
        "kadane|" + cross("contiguous|subarray|sub-array|sub array|slice|sublist|segment",
                          "largest sum|maximum sum|max sum|highest sum|greatest sum|biggest sum|largest&sum"
                          "|maximum&sum|max&sum|highest&sum|greatest&sum"), helpers=(KADANE,))
    window = ("window|k consecutive|consecutive elements|of size k|of length k|sliding|"
              + cross("subarray|sub-array|sublist|slice|block", "of size|of length"))
    windowed = lambda x, k: nums(x) and whole(k, 1, len(x)) and len(x) * k <= 5000  # noqa: E731
    add(2, "window_sums", windowed,
        "[sum({0}[i:i + {1}]) for i in range(len({0}) - {1} + 1)]", window + "|moving sum|rolling sum",
        kind="generic")
    add(2, "max_window_sum", windowed,
        "max(sum({0}[i:i + {1}]) for i in range(len({0}) - {1} + 1))",
        cross(window, "largest|maximum|max|highest|greatest|biggest|best"), kind="generic")
    add(2, "window_means", windowed,
        "[sum({0}[i:i + {1}]) / {1} for i in range(len({0}) - {1} + 1)]",
        "moving average|rolling average|moving mean|rolling mean|" + cross(window, "average|mean"), kind="generic")
    add(1, "prefix_products", lambda x: nums(x, 1, 60), "type({0})(itertools.accumulate({0}, operator.mul))",
        "cumulative product|running product|prefix product|partial product|product&so far|product&up to and including"
        "|products&up to", kind="generic")
    add(1, "prefix_max", nums, "type({0})(itertools.accumulate({0}, max))",
        "running max|cumulative max|prefix max|maximum so far|max so far|largest so far|highest so far|"
        + cross("maximum|largest|greatest|highest", "up to that|up to each|up to and including|so far|until that"))
    add(1, "prefix_min", nums, "type({0})(itertools.accumulate({0}, min))",
        "running min|cumulative min|prefix min|minimum so far|min so far|smallest so far|lowest so far|"
        + cross("minimum|smallest|least|lowest", "up to that|up to each|up to and including|so far|until that"))
    records = "new record|record high|record-break|" + cross(
        "greater than|larger than|bigger than|higher than|exceed",
        "before|previous|preceding|earlier|to their left|to its left")
    leaders = "leader|" + cross(
        "greater than|larger than|bigger than|higher than",
        "to their right|to its right|on its right|on their right|right side|after it|after them")
    ordered = lambda x: nums(x) or strs(x)  # noqa: E731
    add(1, "prefix_maxima", ordered, "_r52_prefix_maxima({0})", records, helpers=(PREFIX_MAXIMA,))
    add(1, "count_prefix_maxima", ordered, "len(_r52_prefix_maxima({0}))", records, helpers=(PREFIX_MAXIMA,))
    add(1, "leaders", ordered, "_r52_leaders({0})", leaders, helpers=(LEADERS,))
    add(1, "count_leaders", ordered, "len(_r52_leaders({0}))", leaders, helpers=(LEADERS,))
    add(1, "alternating_sum", nums, "sum(v if i % 2 == 0 else -v for i, v in enumerate({0}))",
        cross("alternat", "sum|total|add|subtract"))
    add(1, "max_pair_product", lambda x: nums(x, 2),
        "(lambda s: max(s[0] * s[1], s[-1] * s[-2]))(sorted({0}))",
        cross("largest|maximum|max|highest|greatest|biggest", "product", "two|pair"))
    add(1, "count_distinct", lambda x: flat(x, 1, 600) or text(x), "len(set({0}))",
        "distinct|different value|different element|different number|different item|different character"
        "|different letter|unique value|unique element|unique number|unique item|unique character", kind="generic")
    add(1, "above_mean", nums,
        "(lambda m: type({0})(v for v in {0} if v > m))(sum({0}) / len({0}) if len({0}) else 0)",
        cross("above|greater than|more than|larger than|higher than|exceed|bigger than", "average|mean"),
        kind="generic")
    add(1, "count_above_mean", nums, "(lambda m: sum(v > m for v in {0}))(sum({0}) / len({0}) if len({0}) else 0)",
        cross("above|greater than|more than|larger than|higher than|exceed|bigger than", "average|mean"),
        kind="generic")
    add(1, "count_below_mean", nums, "(lambda m: sum(v < m for v in {0}))(sum({0}) / len({0}) if len({0}) else 0)",
        cross("below|less than|smaller than|lower than|under", "average|mean"), kind="generic")
    add(1, "below_mean", nums,
        "(lambda m: type({0})(v for v in {0} if v < m))(sum({0}) / len({0}) if len({0}) else 0)",
        cross("below|less than|smaller than|lower than|under", "average|mean"), kind="generic")
    rank = lambda x: nums(x) or strs(x)  # noqa: E731
    add(1, "dense_rank", rank,
        "(lambda order: type({0})(order[v] for v in {0}))({{v: i + 1 for i, v in enumerate(sorted(set({0})))}})",
        "rank")
    add(1, "competition_rank", rank,
        "(lambda order: type({0})(bisect.bisect_left(order, v) + 1 for v in {0}))(sorted({0}))", "rank")

    # ------------------------------------------------------------ sequences: rearranging and padding
    add(1, "evens_then_odds", ints,
        "type({0})([v for v in {0} if v % 2 == 0] + [v for v in {0} if v % 2 != 0])",
        "even&before&odd|even&first&odd|even&then&odd|even&followed by&odd|even&front&odd|segregat&even"
        "|separat&even&odd|partition&even|evens first|even numbers first", kind="generic")
    add(1, "odds_then_evens", ints,
        "type({0})([v for v in {0} if v % 2 != 0] + [v for v in {0} if v % 2 == 0])",
        "odd&before&even|odd&first&even|odd&then&even|odd&followed by&even|odd&front&even|segregat&odd"
        "|separat&odd&even|partition&odd|odds first|odd numbers first", kind="generic")
    add(1, "negatives_first", nums, "type({0})([v for v in {0} if v < 0] + [v for v in {0} if v >= 0])",
        "segregat&negative|separat&negative&positive|partition&negative|"
        + cross("negative", "before|first|front|followed by|then|beginning|left side"), kind="generic")
    add(1, "positives_first", nums, "type({0})([v for v in {0} if v >= 0] + [v for v in {0} if v < 0])",
        "segregat&positive|separat&positive&negative|partition&positive|"
        + cross("positive", "before|first|front|followed by|then|beginning|left side"), kind="generic")
    # Two conventions for an odd length; the larger first half also carries the words that ask for it.
    halves = lambda x: seq(x, 2) or text(x, 2)  # noqa: E731
    add(1, "halves", halves, "({0}[:len({0}) // 2], {0}[len({0}) // 2:])", "half|halves", kind="generic")
    add(1, "halves_first_longer", halves, "({0}[:(len({0}) + 1) // 2], {0}[(len({0}) + 1) // 2:])",
        "half|halves|" + cross("first half", "extra|longer|larger|bigger|more|one more"), kind="generic")
    add(1, "consecutive_runs", ints, "_r52_consecutive_runs({0})",
        "runs of consecutive|consecutive runs|consecutive ranges|summary range|"
        + cross("split|group|break|partition|divide|separate", "consecutive"), helpers=(CONSECUTIVE_RUNS,))
    add(2, "hamming_distance", lambda a, b: (text(a, 0) and text(b, 0) or flat(a) and flat(b)) and len(a) == len(b),
        "sum(x != y for x, y in zip({0}, {1}))",
        "hamming|mismatch|" + cross("position|index|indices|place", "differ|mismatch|not equal|unequal"))
    add(2, "ordered_common", lambda x, y: flat(x) and flat(y),
        "(lambda other: type({0})(dict.fromkeys(v for v in {0} if v in other)))(set({1}))",
        cross("common|both|intersection|shared|present in|appear in|occur in",
              "order|repeat|duplicate|unique|first appear|once|distinct"), kind="generic")
    def padding(x, n, fill):
        return is_seq(x) and len(x) <= 300 and whole(n, 0, 1000) and (scalar(fill) or fill is None)
    add(3, "pad_right", padding, "{0}[:] + type({0})([{2}]) * max(0, {1} - len({0}))", "pad", kind="generic")
    add(3, "pad_left", padding, "type({0})([{2}]) * max(0, {1} - len({0})) + {0}[:]", "pad", kind="generic")
    justify = lambda s, w, c: text(s, 0) and whole(w, 0, 1000) and text(c, 1, 1)  # noqa: E731
    add(3, "ljust", justify, "{0}.ljust({1}, {2})", "pad|justif|align|fill", kind="generic")
    add(3, "rjust", justify, "{0}.rjust({1}, {2})", "pad|justif|align|fill", kind="generic")
    centre = "center|centre|centred|centered"
    add(3, "center", justify, "{0}.center({1}, {2})", centre, kind="generic")
    add(3, "center_right_extra", justify, "_r52_center({0}, {1}, {2})", centre, helpers=(CENTER,), kind="generic")

    # ------------------------------------------------------------ strings: case and word conventions
    add(1, "snake_to_lower_camel", lambda s: text(s, 1, 200) and " " not in s,
        "{0}.split('_')[0].lower() + ''.join(w.capitalize() for w in {0}.split('_')[1:])", "camel", kind="generic")
    add(1, "to_kebab", text, "'-'.join({0}.lower().split())",
        "kebab|slug|hyphen-separated|dash-separated|hyphenated|" + cross("hyphen|dash", "lower"), kind="generic")
    add(1, "alternate_case_upper_first", text, "''.join(c.upper() if i % 2 == 0 else c.lower() "
        "for i, c in enumerate({0}))", "alternat&case|alternat&upper|alternat&capital|alternat&lower|zigzag|sponge|"
        + cross("even ind|odd ind|even position|odd position", "upper|capital|lower"), kind="generic")
    add(1, "alternate_case_lower_first", text, "''.join(c.lower() if i % 2 == 0 else c.upper() "
        "for i, c in enumerate({0}))", "alternat&case|alternat&upper|alternat&capital|alternat&lower|zigzag|sponge|"
        + cross("even ind|odd ind|even position|odd position", "upper|capital|lower"), kind="generic")
    add(1, "vowels_upper", text, "''.join(c.upper() if c.lower() in 'aeiou' else c.lower() for c in {0})",
        cross("vowel", "upper|capital"), kind="generic")
    words = lambda s: text(s) and any(c.isalpha() for c in s)  # noqa: E731
    add(1, "initials_dotted", words, "''.join(w[0].upper() + '.' for w in {0}.split())", "initial", kind="generic")
    add(1, "initials_dotted_spaced", words, "' '.join(w[0].upper() + '.' for w in {0}.split())", "initial",
        kind="generic")
    add(1, "middle_chars", text, "{0}[(len({0}) - 1) // 2:len({0}) // 2 + 1]", "middle", kind="generic")
    add(1, "sort_words_by_length", lambda s: text(s) and " " in s.strip(), "' '.join(sorted({0}.split(), key=len))",
        cross("word", "length", "sort|order|arrang|increasing|ascending|shortest"), kind="generic")
    add(2, "words_starting_with", lambda s, p: text(s) and text(p, 1, 30),
        "[w for w in {0}.split() if w.startswith({1})]", cross("word", "start|begin"), kind="generic")
    add(2, "words_starting_with_any_case", lambda s, p: text(s) and text(p, 1, 30),
        "[w for w in {0}.split() if w.lower().startswith({1}.lower())]",
        cross("word", "start|begin", "ignor|case-insensitive|case insensitive|regardless of case|either case|any case"
                                     "|upper or lower|lower or upper"), kind="generic")
    add(1, "word_lengths_dict", lambda s: text(s), "{{w: len(w) for w in {0}.split()}}",
        cross("word", "length", "dict|map|key"), kind="generic")
    # Per-word counts only when the description asks for a count per word: "count the words" wants one number.
    per_word = cross("word", "frequen|occurr|how many times|number of times|tally|each word|every word")
    add(1, "word_counts", lambda s: text(s), "dict(collections.Counter({0}.split()))", per_word, kind="generic")
    add(1, "word_counts_lower", lambda s: text(s), "dict(collections.Counter({0}.lower().split()))",
        cross(per_word, "lower|case"), kind="generic")
    add(1, "join_and", lambda x: strs(x, 1, 50), "', '.join({0}[:-1]) + ' and ' + {0}[-1] if len({0}) > 1 "
        "else ''.join({0})", "' and '|before the last|and&last word|english list|human-readable list"
                             "|natural language list", kind="generic")
    add(1, "join_and_oxford", lambda x: strs(x, 1, 50), "', '.join({0}[:-1]) + ', and ' + {0}[-1] if len({0}) > 2 "
        "else ' and '.join({0})", "' and '|before the last|and&last word|english list|human-readable list"
                                  "|natural language list|oxford comma|serial comma", kind="generic")

    # ------------------------------------------------------------ strings: encodings and ciphers
    rle = "run-length|run length|rle|compress"
    add(1, "rle_char_count", text, "''.join(k + str(len(list(g))) for k, g in itertools.groupby({0}))", rle)
    add(1, "rle_count_char", text, "''.join(str(len(list(g))) + k for k, g in itertools.groupby({0}))", rle)
    decode = "decod|decompress|uncompress|expand"
    add(1, "rle_decode_char_count",
        lambda s: text(s, 2) and scope["re"].fullmatch(r"(\D\d+)+", s) is not None,
        r"''.join(c * int(n) for c, n in re.findall(r'(\D)(\d+)', {0}))", decode)
    add(1, "rle_decode_count_char",
        lambda s: text(s, 2) and scope["re"].fullmatch(r"(\d+\D)+", s) is not None,
        r"''.join(c * int(n) for n, c in re.findall(r'(\d+)(\D)', {0}))", decode)
    shift = lambda s, k: text(s) and is_int(k) and abs(k) <= 10 ** 6  # noqa: E731
    add(2, "caesar", shift, "_r52_caesar({0}, {1})",
        "caesar|cipher|encrypt|" + cross("shift|rotat|move", "letter|alphabet"), helpers=(CAESAR,))
    add(2, "caesar_decode", shift, "_r52_caesar({0}, -{1})",
        cross("caesar|cipher|shift", "decod|decrypt|backward|back by"), helpers=(CAESAR,))
    add(1, "rot13", text, "_r52_caesar({0}, 13)", "rot13|rot-13|rot 13", helpers=(CAESAR,))
    add(1, "atbash", text, "_r52_atbash({0})",
        "atbash|reverse alphabet|reversed alphabet|mirror&alphabet|a with z|a becomes z", helpers=(ATBASH,))
    add(1, "pig_latin", lambda s: text(s, 1, 60) and s.isalpha(), "_r52_pig_latin({0})", "pig latin|pig-latin",
        helpers=(PIG_LATIN,))
    add(1, "numeronym", lambda s: text(s, 1, 100) and s.isalpha(), "{0}[0] + str(len({0}) - 2) + {0}[-1] "
        "if len({0}) > 2 else {0}", "numeronym|abbreviat")
    add(2, "numeronym_long_words", lambda s, n: text(s) and " " in s and whole(n, 2, 50),
        "' '.join(w[0] + str(len(w) - 2) + w[-1] if len(w) > {1} else w for w in {0}.split(' '))",
        "numeronym|abbreviat")
    add(1, "to_12_hour", hhmm, "_r52_to_12_hour({0})",
        "12-hour|12 hour|twelve-hour|twelve hour|am/pm|am or pm|a.m.|'am'|'pm'", helpers=(TWELVE_HOUR,))
    add(1, "to_24_hour", lambda t: text(t, 4, 14) and scope["re"].fullmatch(
        r"\s*\d{1,2}:\d{2}(:\d{2})?\s*[AaPp]\.?[Mm]\.?\s*", t) is not None, "_r52_to_24_hour({0})",
        "24-hour|24 hour|twenty-four hour|twenty four hour|military time|24h", helpers=(TWENTY_FOUR_HOUR,))
    add(1, "to_12_hour_padded", hhmm, "_r52_to_12_hour({0}, True)",
        "12-hour|12 hour|twelve-hour|twelve hour|am/pm|am or pm|a.m.|'am'|'pm'", helpers=(TWELVE_HOUR,))

    # ------------------------------------------------------------ strings: editing
    add(2, "interleave", lambda a, b: (text(a, 0) and text(b, 0) and len(a) + len(b) > 0)
        or (is_seq(a) and is_seq(b) and type(a) is type(b) and 0 < len(a) + len(b) <= 300),
        "_r52_interleave({0}, {1})", "interleav|alternat|one from each|take turns|in turn", helpers=(INTERLEAVE,),
        kind="generic")
    add(2, "repeat_each_char", lambda s, n: text(s, 1, 200) and whole(n, 0, 20), "''.join(c * {1} for c in {0})",
        cross("repeat|duplicate|copies", "each char|every char|each letter|every letter"), kind="generic")
    add(1, "double_each_char", lambda s: text(s, 1, 300), "''.join(c * 2 for c in {0})",
        cross("twice|double", "char|letter") + "|doubled", kind="generic")
    add(2, "replace_digits", lambda s, c: text(s) and text(c, 0, 5),
        "''.join({1} if c.isdigit() else c for c in {0})",
        cross("replace|substitut|mask|hide|change|swap|convert", "digit|number"), kind="generic")
    add(2, "replace_vowels", lambda s, c: text(s) and text(c, 0, 5),
        "''.join({1} if c.lower() in 'aeiou' else c for c in {0})", cross("replace|substitut|mask|change", "vowel"),
        kind="generic")
    mask = "mask|hide|obscure|conceal|redact"
    add(3, "mask_all_but_last", lambda s, n, c: text(s, 1, 300) and whole(n, 0, 300) and text(c, 1, 1),
        "{2} * max(0, len({0}) - {1}) + {0}[len({0}) - min({1}, len({0})):]", mask, kind="generic")
    add(2, "mask_all_but_last_four", lambda s, c: text(s, 1, 300) and text(c, 1, 1),
        "{1} * max(0, len({0}) - 4) + {0}[-4:]", cross(mask, "last four|last 4|final four|final 4"), kind="generic")
    shorten = "shorten|truncat|ellipsis|abbreviat|cut off"
    truncation = lambda s, n, end: text(s) and whole(n, 0, 600) and text(end, 1, 5)  # noqa: E731
    add(3, "truncate", truncation, "{0} if len({0}) <= {1} else {0}[:{1}] + {2}", shorten, kind="generic")
    add(3, "truncate_to_total", truncation, "{0} if len({0}) <= {1} else {0}[:max(0, {1} - len({2}))] + {2}", shorten,
        kind="generic")
    add(1, "strip_tags", lambda s: text(s) and "<" in s, r"re.sub(r'<[^>]*>', '', {0})",
        "html|xml|tags|markup|angle bracket", kind="generic")
    add(1, "hashtags", lambda s: text(s) and "#" in s, r"re.findall(r'#\w+', {0})", "hashtag|hash tag")

    # ------------------------------------------------------------ predicates on numbers
    natural = lambda n: whole(n, 0, 10 ** 18)  # noqa: E731
    predicate(1, "is_automorphic", natural, "str({0} * {0}).endswith(str({0}))", "automorphic")
    predicate(1, "is_factorion", natural, "{0} > 0 and {0} == sum(math.factorial(int(d)) for d in str({0}))",
              "factorion|strong number|krishnamurthy|peterson number|sum of the factorials|sum of factorials"
              "|factorials of its digits|factorial of each digit|factorial of its digits")
    predicate(1, "is_perfect_cube", lambda n: is_int(n) and abs(n) <= 10 ** 30, "_r52_is_cube({0})",
              "perfect cube|cube number|is a cube|cube of an integer|cube of some integer|k * k * k|cubic number",
              helpers=(IS_CUBE,))
    predicate(1, "is_self_dividing", natural, "{0} > 0 and all(d != '0' and {0} % int(d) == 0 for d in str({0}))",
              "self-dividing|self dividing|" + cross(
                  "divisible by", "each of its digits|every digit|all of its digits|all its digits|each digit"
                                  "|its digits|each of its own digits|all the digits"))
    predicate(1, "is_ugly", lambda n: whole(n, 0, 10 ** 18), "_r52_is_ugly({0})",
              "ugly|hamming number|regular number|5-smooth|only prime factors|no prime factors other than"
              "|2, 3 and 5|2, 3, and 5|2, 3 or 5|2, 3, or 5", helpers=(IS_UGLY,))
    # One rule per common base: the engine reads digits from the description, not number words.
    for base, word in ((3, "three"), (4, "four"), (5, "five"), (10, "ten")):
        predicate(1, f"is_power_of_{base}", lambda n: whole(n, 1, 10 ** 30), f"_r52_is_power_of({{0}}, {base})",
                  f"power of {word}|power of {base}", helpers=(POWER_OF,))
    predicate(1, "is_binary_palindrome", natural, "bin({0})[2:] == bin({0})[2:][::-1]",
              cross("binary", "palindrom|reads the same|same forward|same backward"))
    for parity, test in (("even", "== 0"), ("odd", "== 1")):
        predicate(1, f"all_digits_{parity}", natural, f"all(int(d) % 2 {test} for d in str(abs({{0}})))",
                  cross("every digit|all digits|all of its digits|all its digits|each digit|all the digits"
                        f"|only {parity} digits", parity))

    # ------------------------------------------------------------ predicates on strings
    predicate(1, "is_identifier", text, r"re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', {0}) is not None",
              "identifier|variable name")
    predicate(1, "is_hex_colour", text, r"re.fullmatch(r'#[0-9A-Fa-f]{{6}}', {0}) is not None",
              "hex colo|colour code|color code|html colo|hex code|hexadecimal colo|css colo")
    predicate(1, "is_ipv4", text, "_r52_is_ipv4({0})", "ipv4|ip address|ip-address|ip addr", helpers=(IS_IPV4,))
    predicate(1, "is_binary_string", lambda s: text(s, 0), "len({0}) > 0 and set({0}) <= {{'0', '1'}}",
              "binary string|bit string|binary number|valid binary|binary digits|0s and 1s|zeros and ones"
              "|'0' and '1'|only of 0|only 0 and 1")
    predicate(1, "is_time_24h", text, r"re.fullmatch(r'([01]\d|2[0-3]):[0-5]\d', {0}) is not None",
              "24-hour|24 hour|hh:mm|valid time|clock time|military time")
    predicate(1, "is_email", text, r"re.fullmatch(r'[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+\.)+[A-Za-z]{{2,6}}', {0}) "
              "is not None", "email|e-mail")
    predicate(1, "is_phone_dashed", text, r"re.fullmatch(r'\d{{3}}-\d{{3}}-\d{{4}}', {0}) is not None", "phone")
    predicate(1, "is_clean_palindrome", text, "(lambda t: t == t[::-1])([c.lower() for c in {0} if c.isalnum()])",
              cross("palindrom|reads the same|same forward|same backward|same in reverse|same when reversed",
                    "ignor|non-alphanumeric|punctuation|letter case|regardless|only letters|only alphanumeric"
                    "|case-insensitive|case insensitive|spaces"))
    predicate(1, "is_isogram", text, "(lambda t: len(t) == len(set(t)))([c.lower() for c in {0} if c.isalpha()])",
              "isogram|no letter&more than once|no letter&repeat|no letter is repeated|repeating letter"
              "|repeated letter|each letter&only once|" + cross(
                  "unique|distinct|no duplicate|not repeat|no repeat",
                  "ignoring case|ignore case|ignores case|case-insensitive|case insensitive|regardless of case"))
    predicate(1, "is_question", lambda s: text(s, 0), "{0}.rstrip().endswith('?')", "question mark")
    predicate(2, "anagram_ignoring_case_and_spaces", lambda a, b: text(a) and text(b),
              "sorted(c for c in {0}.lower() if c.isalnum()) == sorted(c for c in {1}.lower() if c.isalnum())",
              cross("anagram", "ignor|space|case|capital|punctuation|regardless|whitespace|insensitive"))
    insensitive = "case|ignor|insensitive|regardless"
    predicate(2, "equal_ignoring_case", lambda a, b: text(a, 0) and text(b, 0), "{0}.lower() == {1}.lower()",
              cross("equal|same|identical|match", "ignor&case|case-insensitive|case insensitive|regardless of case"))
    predicate(2, "ends_with_any_case", lambda s, t: text(s, 0) and text(t, 0), "{0}.lower().endswith({1}.lower())",
              cross("ends with|end with|ending with|suffix", insensitive))
    predicate(2, "starts_with_any_case", lambda s, t: text(s, 0) and text(t, 0),
              "{0}.lower().startswith({1}.lower())",
              cross("starts with|start with|begins with|begin with|prefix", insensitive))
    password = lambda p, n: text(p, 0, 200) and whole(n, 1, 100)  # noqa: E731
    # The full rule outranks the round-3 "one letter and one digit" predicate that the same wording names.
    strong = "password|" + cross("password", "upper|capital", "lower", "digit|number")
    predicate(2, "strong_password", password,
              "len({0}) >= {1} and any(c.isupper() for c in {0}) and any(c.islower() for c in {0}) "
              "and any(c.isdigit() for c in {0})", strong)
    predicate(2, "strong_password_with_symbol", password,
              "len({0}) >= {1} and any(c.isupper() for c in {0}) and any(c.islower() for c in {0}) "
              "and any(c.isdigit() for c in {0}) and any(not c.isalnum() for c in {0})",
              strong + "|" + cross("password", "special|symbol|punctuation|non-alphanumeric"))

    # ------------------------------------------------------------ predicates on sequences
    several = lambda x: nums(x, 1)  # noqa: E731
    predicate(1, "is_arithmetic", several, "all(b - a == {0}[1] - {0}[0] for a, b in zip({0}, {0}[1:]))",
              "arithmetic|same difference|common difference|constant difference|equal difference|evenly spaced")
    predicate(1, "is_geometric", several, "all(b * {0}[0] == a * {0}[1] for a, b in zip({0}, {0}[1:]))",
              "geometric|same ratio|common ratio|constant ratio|equal ratio")
    predicate(1, "alternating_parity", lambda x: ints(x), "all(a % 2 != b % 2 for a, b in zip({0}, {0}[1:]))",
              cross("alternat", "even|odd|parity"))
    predicate(1, "is_mountain", several, "_r52_is_mountain({0})",
              "mountain|single peak|one peak|increases and then decreases|increases then decreases"
              "|increasing then decreasing|increasing and then decreasing", helpers=(MOUNTAIN,))
    predicate(1, "is_permutation_of_1_to_n", lambda x: ints(x), "sorted({0}) == list(range(1, len({0}) + 1))",
              "permutation of 1|permutation&1 to n|1 to n&exactly once|from 1 to n&once|each integer from 1"
              "|each number from 1|every integer from 1|numbers 1 to n|integers 1 to n")
    # Named with "two" or "any": "all consecutive elements are equal" is the round-3 all_same.
    predicate(1, "has_adjacent_equal", lambda x: flat(x) or text(x, 0), "any(a == b for a, b in zip({0}, {0}[1:]))",
              "next to each other|side by side|double letter|doubled letter|"
              + cross("adjacent|consecutive", "identical|equal|same|repeated|duplicate", "two|pair|any"))
    predicate(2, "has_pair_with_sum", lambda x, t: nums(x) and number(t), "_r52_has_pair_sum({0}, {1})",
              cross("pair|two numbers|two elements|two different|two distinct|two values|two positions",
                    "sum|add up|adds up|total"), helpers=(PAIR_SUM,))
    square = lambda m: matrix(m) and len(m) == len(m[0])  # noqa: E731
    predicate(1, "is_identity_matrix", square,
              "all(v == (i == j) for i, row in enumerate({0}) for j, v in enumerate(row))", "identity")

    # ------------------------------------------------------------ nested: matrices
    add(1, "identity_matrix", lambda n: whole(n, 0, 30), "[[int(i == j) for j in range({0})] for i in range({0})]",
        "identity")
    add(1, "times_table", lambda n: whole(n, 1, 30),
        "[[i * j for j in range(1, {0} + 1)] for i in range(1, {0} + 1)]",
        "multiplication table|times table|multiplication grid|multiplication matrix")
    add(1, "pascal_rows", lambda n: whole(n, 0, 30), "[[math.comb(i, j) for j in range(i + 1)] for i in range({0})]",
        "pascal")
    add(1, "main_diagonal", lambda m: matrix(m, lambda v: True),
        "[{0}[i][i] for i in range(min(len({0}), len({0}[0])))]", "diagonal|trace")
    add(1, "trace", lambda m: matrix(m), "sum({0}[i][i] for i in range(min(len({0}), len({0}[0]))))",
        "trace|" + cross("sum|total|add", "diagonal"))
    # "Anti-diagonal" contains "diagonal": the longer words make the anti-diagonal rules win where both are named.
    anti = ("anti-diagonal|anti diagonal|antidiagonal|secondary diagonal|other diagonal|second diagonal"
            "|minor diagonal|counter-diagonal|counter diagonal")
    add(1, "anti_diagonal", lambda m: matrix(m, lambda v: True),
        "[{0}[i][len({0}[i]) - 1 - i] for i in range(min(len({0}), len({0}[0])))]", anti)
    add(1, "anti_trace", lambda m: matrix(m),
        "sum({0}[i][len({0}[i]) - 1 - i] for i in range(min(len({0}), len({0}[0]))))", cross("sum|total|add", anti))
    add(1, "spiral_order", lambda m: matrix(m, lambda v: True), "_r52_spiral_order({0})", "spiral",
        helpers=(SPIRAL,))
    # A matrix of two rows is all border: the gate avoids "outer list" wording, and the rule stays generic so a
    # named flatten wins the tie on such an example.
    add(1, "border_elements", lambda m: matrix(m), "_r52_border({0})",
        "border|boundary|outermost|outer ring|outer layer|outer edge|perimeter&matrix|edge&matrix", helpers=(BORDER,),
        kind="generic")
    add(1, "border_clockwise", lambda m: matrix(m), "_r52_border_clockwise({0})",
        "border|boundary|outermost|outer ring|outer layer|outer edge|perimeter&matrix|edge&matrix",
        helpers=(BORDER_CLOCKWISE,), kind="generic")
    add(2, "matrix_product", lambda a, b: matrix(a) and matrix(b) and len(a[0]) == len(b)
        and len(a) * len(b) * len(b[0]) <= 3000,
        "[[sum(x * y for x, y in zip(row, col)) for col in zip(*{1})] for row in {0}]",
        "matmul|" + cross("multipl|product", "matri"))
    add(2, "grid_positions", lambda m, v: seqseq(m) and sum(map(len, m)) <= 900 and scalar(v),
        "[(r, c) for r, row in enumerate({0}) for c, e in enumerate(row) if e == {1}]",
        cross("position|index|indices|coordinate|location|where|row and column",
              "matrix|grid|2d|2-d|list of lists"), kind="generic")
    rows_of_numbers = lambda m: seqseq(m) and 2 <= len(m) <= 300 and all(nums(r, 0) for r in m)  # noqa: E731
    largest_sum = "largest sum|maximum sum|highest sum|greatest sum|max sum|biggest sum|largest total|highest total"
    add(1, "row_with_largest_sum", rows_of_numbers, "max({0}, key=sum)",
        cross("list|row|sublist|sub-list|inner", largest_sum), kind="generic")
    add(1, "index_of_row_with_largest_sum", rows_of_numbers, "max(range(len({0})), key=lambda i: sum({0}[i]))",
        cross("index|position|which", "list|row|sublist|sub-list|inner", largest_sum), kind="generic")
    add(1, "nested_sum", lambda m: seqseq(m) and len(m) <= 300 and all(nums(r, 0) for r in m) and any(m),
        "sum(v for row in {0} for v in row)",
        "sum of all|total of all|add up all|sum&nested|sum&matrix|sum&list of lists|sum&grid|sum&2d|sum&every"
        "|total&nested|total&matrix|sum&inner list|sum&sublists|" + cross("sum|total", "all elements|all numbers"
                                                                          "|all values|list values"), kind="generic")
    add(1, "unzip_pairs", lambda x: seq(x) and all(is_seq(p) and len(p) == 2 for p in x),
        "([p[0] for p in {0}], [p[1] for p in {0}])",
        "unzip|two lists|first items&second items|first elements&second elements|"
        + cross("split|separate|divide|break", "pairs|tuples"), kind="generic")
    add(2, "replace_none", lambda x, v: seq(x) and scalar(v) and any(
        e is None or (is_seq(e) and any(f is None for f in e)) for e in x),
        "type({0})(type(row)({1} if e is None else e for e in row) if isinstance(row, (list, tuple)) "
        "else ({1} if row is None else row) for row in {0})",
        cross("none|null|missing", "replace|fill|substitut|default|convert"), kind="generic")
    add(2, "cartesian_product", lambda x, y: seq(x, 1, 30) and seq(y, 1, 30), "list(itertools.product({0}, {1}))",
        "every pair|all pairs|cartesian|all possible pairs|every combination|each pair|all ordered pairs"
        "|pairs (a, b)|pair (a, b)", kind="generic")

    # ------------------------------------------------------------ nested: grouping and dictionaries
    group = "group|categori|bucket|dictionary|keyed|map"
    add(1, "group_by_initial", lambda x: strs(x), "_r52_group_by({0}, lambda w: w[:1])",
        cross(group, "first letter|first character|first char|initial|starting letter|starting character"),
        helpers=(GROUP_BY,), kind="generic")
    add(1, "group_by_length", lambda x: seq(x) and all(isinstance(v, (str, list, tuple)) for v in x),
        "_r52_group_by({0}, len)", cross(group, "length"), helpers=(GROUP_BY,), kind="generic")
    add(2, "group_by_remainder", lambda x, k: ints(x) and whole(k, 1, 10 ** 6),
        "_r52_group_by({0}, lambda v: v % {1})", cross(group, "remainder|modulo|mod "), helpers=(GROUP_BY,),
        kind="generic")
    add(1, "group_anagrams", lambda x: strs(x), "list(_r52_group_by({0}, lambda w: ''.join(sorted(w))).values())",
        cross("group|categori|bucket|cluster", "anagram"), helpers=(GROUP_BY,), kind="generic")
    add(1, "sum_by_key", lambda x: seq(x) and all(is_seq(p) and len(p) == 2 and is_num(p[1]) for p in x),
        "_r52_sum_by_key({0})", cross("sum|total|add|combin|aggregat|accumulat", "key|same name|same first|per "),
        helpers=(SUM_BY_KEY,), kind="generic")
    valued = lambda d: isinstance(d, dict) and 0 < len(d) <= 200  # noqa: E731
    comparable = lambda d: valued(d) and (all(is_num(v) for v in d.values())  # noqa: E731
                                          or all(isinstance(v, str) for v in d.values()))
    add(1, "keys_by_value_desc", comparable, "sorted({0}, key={0}.get, reverse=True)",
        cross("key", "value", "descend|decreas|highest first|largest first|high to low|largest to smallest"
                              "|highest to lowest|reverse"), kind="generic")
    add(1, "keys_by_value", comparable, "sorted({0}, key={0}.get)",
        cross("key", "value", "ascend|increas|lowest first|smallest first|low to high|smallest to largest|sort"),
        kind="generic")
    listed = lambda d: valued(d) and all(isinstance(v, (list, tuple)) for v in d.values())  # noqa: E731
    numbers_listed = lambda d: listed(d) and all(is_num(x) for v in d.values() for x in v)  # noqa: E731
    per_key = "each key|value|list|per |each name|each student|each person"
    add(1, "dict_value_lengths", listed, "{{k: len(v) for k, v in {0}.items()}}",
        cross("length|count|size|how many|number of", per_key), kind="generic")
    add(1, "dict_value_sums", numbers_listed, "{{k: sum(v) for k, v in {0}.items()}}", cross("sum|total", per_key),
        kind="generic")
    means = lambda d: numbers_listed(d) and all(d.values())  # noqa: E731
    add(1, "dict_value_means", means, "{{k: sum(v) / len(v) for k, v in {0}.items()}}",
        cross("average|mean", per_key), kind="generic")
    for places, phrases in PLACES.items():
        add(1, f"dict_value_means_r{places}", means, f"{{{{k: round(sum(v) / len(v), {places}) for k, v in "
            f"{{0}}.items()}}}}", cross("average|mean", per_key, phrases), kind="generic")
    add(1, "merge_dicts_in_list", lambda x: seq(x, 1, 100) and all(isinstance(d, dict) for d in x),
        "{{k: v for d in {0} for k, v in d.items()}}", cross("merge|combin|union|join|single|one dict", "dict"),
        kind="generic")
    add(2, "get_by_path", lambda d, keys: isinstance(d, dict) and seq(keys, 1, 20) and all(scalar(k) for k in keys),
        "_r52_get_path({0}, {1})",
        "path|chain of keys|sequence of keys|list of keys|series of keys|nested&key|follow&key|keys&nested",
        helpers=(GET_PATH,), kind="generic")
    add(1, "count_leaf_values", lambda d: isinstance(d, dict), "_r52_count_leaves({0})",
        "leaf|leaves|not themselves dict|non-dict|" + cross("count", "nested dict|nested&dictionar"),
        helpers=(COUNT_LEAVES,))

    def numeric_tree(d, depth=0):
        return depth < 20 and isinstance(d, dict) and all(
            numeric_tree(v, depth + 1) if isinstance(v, dict) else is_num(v) for v in d.values())
    add(1, "nested_dict_sum", numeric_tree, "_r52_nested_dict_sum({0})",
        cross("sum|total|add", "nested dict|nested&dictionar|anywhere"), helpers=(NESTED_DICT_SUM,))

    # ------------------------------------------------------------ more words for existing operators
    # Registered before the rounded forms below, which copy the words of the rule they round.
    # A leap-year description usually spells out the divisibility rule, which also names "divisible by 4".
    also("is_leap_year/1", cross("leap year|leap-year", "divisible by 4|divisible by 100|divisible by 400|gregorian"
                                 "|century"))
    also("num_lcm_upto/1", "divisible by every|divisible by each|divisible by all the|divisible by all integers"
         "|divisible by all numbers|divisible by 1 to|divisible by all of the")
    also("num_highest_power_of_2_upto/1", "biggest power of two|biggest power of 2|down to a power of two"
         "|down to a power of 2|" + cross("power of two|power of 2", "not exceed|does not exceed|at most"))
    also("mersenne/1", "hanoi")
    also("median3/3", "middle&three|middle value|middle number|middle of three|middle one")
    also("num_harmonic_sum/1", "1 + 1/2|1/2 + 1/3|reciprocal")
    also("num_hamming_distance/2", "bit position&differ|binary representation&differ|bits that differ|differing bit")
    also("seq_count_pairs_sum/2", "count&pair&add up|number of&pair&add up|pairs&add up to|count&pairs&target")
    also("seq_first_missing_positive/1", "missing integer|missing&1 to n|missing&from 1|missing number")
    also("is_pangram/1", "every letter of the alphabet|every letter of the english alphabet|all the letters of the"
         "|all letters of the alphabet|each letter of the alphabet|all 26 letters")
    also("all_unique/1", "no value appears more than once|no element appears more than once|no item appears more than"
         "|no number appears more than once|no repeated value|no repeated element|without repetition")
    also("only_from/2", "every element of the first|all elements of the first|each element of the first"
         "|every item of the first|all items of the first|is a subset|subset of")
    also("is_rotation/2", "obtained by rotating|by rotating|rotating the first|rotated version|rotation of the first"
         "|rotation of the second|cyclic shift|circular shift|cyclic rotation")
    also("coprime/2", "no common divisor|no common factor|common divisor other than 1|common factor other than 1"
         "|gcd is 1|only common divisor|only common factor|share no common")
    also("are_disjoint/2", "no element in common|no elements in common|no common|share no|no shared|no overlap")
    also("is_fibonacci_number/1", "belong&fibonacci|in the fibonacci sequence|part of the fibonacci"
         "|member of the fibonacci|appears in the fibonacci|in fibonacci")
    also("is_palindrome/1", "same forwards and backwards|same backward|same forward|reads the same"
         "|equal to its own reverse|equal to its reverse|same from both ends|same in reverse|same when reversed")
    also("is_balanced/1", "bracket&match|bracket&closed|brackets&correct order|properly nested|well-formed"
         "|well formed|parenthes&closed|matching bracket")
    also("txt_ord_each/1", "by ord|ord(|character code|char code|ord value|ord of")
    also("num_last_digit/1", "last decimal digit|final digit")

    # ------------------------------------------------------------ rounded forms of existing formulas
    for key in ROUNDED_EXISTING:
        name, arity = key.rsplit("/", 1)
        base = next((op for op in {"1": lib.U, "2": lib.B, "3": lib.T}[arity] if op.name == name), None)
        if base is not None and not base.template.startswith("ELEMENTWISE:"):
            words = "|".join("&".join(alternative) for alternative in base.alternatives)
            rounded(int(arity), name, base.fn, base.template, words, base.kind, base.helper)
