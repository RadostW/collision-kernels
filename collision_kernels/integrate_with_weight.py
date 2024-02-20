import numpy as np


def integrate_with_weight(weight, sampler, absolute_tolerance, max_step_size):
    """
    Compute integral of the form

    :math:`\\int_0^\\infty weight(x) \mathbb{sampler(x)} dx`

    Parameters
    ----------
    weight : callable
        weight function. Signature float -> float

    sampler : callable
        probablility source. Signature float x, int n -> list([0,1])

    absolute_tolerance : float
        smaller value - more precise result

    max_step_size : float
        largest step on the transformed integral

    Returns
    -------
    float
        value of the integral

    """

    def weight_transformed(h):
        if h == 1:
            return 0
        r = h / (1 - h)
        dr_over_dh = 1 / (1 - h) ** 2
        return weight(r) * dr_over_dh

    def sampler_transformed(h, n):
        if h == 1:
            return 0
        r = h / (1 - h)
        return sampler(r, n)

    return _integrate_on_interval(
        weight_transformed, sampler_transformed, absolute_tolerance, max_step_size
    )


def _integrate_on_interval(weight, sampler, absolute_tolerance, max_step_size):
    """
    Compute integral of the form

    :math:`\\int_0^1 weight(x) \mathbb{sampler(x)} dx`

    Parameters
    ----------
    weight : callable
        weight function. Signature float -> float

    sampler : callable
        probablility source. Signature float x, int n -> list([0,1])

    absolute_tolerance : float
        smaller value - more precise result

    max_step_size : float
        largest step on the transformed integral

    Returns
    -------
    float
        value of the integral

    """

    initial_n = 100
    max_retries = 10

    def fun(x, _absolute_tolerance):

        tolerable_variance = _absolute_tolerance

        if x == 1:
            return 0

        mean_value = np.mean(sampler(x, initial_n))
        weight_value = weight(x)

        sigma = weight_value * ((mean_value * (1 - mean_value) / initial_n) ** 0.5)

        n_to_add = 0
        total_n = initial_n

        # draw more samples if needed
        for i in range(max_retries):
            # print(f"Retrying {i} {total_n=} {sigma**2}")
            if sigma**2 > tolerable_variance:
                n_to_add = max(
                    int(
                        weight_value**2
                        * mean_value
                        * (1 - mean_value)
                        / tolerable_variance
                    )
                    + 1
                    - total_n,
                    0,
                )
                if n_to_add > 0:
                    better_mean = np.mean(sampler(x, n_to_add))
                    mean_value = (mean_value * total_n + better_mean * n_to_add) / (
                        total_n + n_to_add
                    )
                    total_n = total_n + n_to_add
                    sigma = weight_value * (
                        (mean_value * (1 - mean_value) / total_n) ** 0.5
                    )
            else:
                break

        print(f"Transformed: {x:5.4f} {weight(x):5.4f} {mean_value:5.4f} {total_n} {_absolute_tolerance:5.4e}")

        return weight(x) * mean_value

    return integrate_with_adaptive_simpson(
        fun, 0, 1, absolute_tolerance=absolute_tolerance, max_step_size=max_step_size
    )


def _evaluate_simpsons_rule(f_values, f, a, b, absolute_tolerance):
    """
    Evaluate Simpson's Rule using cached function values.
    """
    m = (a + b) / 2
    fm = f(m, absolute_tolerance / abs(b - a))
    f_values[m] = fm
    simp = abs(b - a) / 6 * (f_values[a] + 4 * fm + f_values[b])
    return m, fm, simp


def _asr(f_values, f, a, b, absolute_tolerance, max_step_size):
    """
    Efficient recursive implementation of adaptive Simpson's rule using cached function values.
    """
    m = (a + b) / 2
    fa, fb, fm = f_values[a], f_values[b], f_values[m]

    left_m, left_fm, left_simp = _evaluate_simpsons_rule(
        f_values, f, a, m, absolute_tolerance
    )
    right_m, right_fm, right_simp = _evaluate_simpsons_rule(
        f_values, f, m, b, absolute_tolerance
    )
    delta = left_simp + right_simp - (fa + fb) * (b - a) / 2

    if abs(delta) <= absolute_tolerance * 15 and abs(b - a) < max_step_size:
        return left_simp + right_simp + delta / 15
    elif abs(b-a) < 1e-1:
        return left_simp + right_simp + delta / 15
    else:
        return _asr(
            f_values,
            f,
            a,
            m,
            absolute_tolerance / 2,
            max_step_size,
        ) + _asr(
            f_values,
            f,
            m,
            b,
            absolute_tolerance / 2,
            max_step_size,
        )


def integrate_with_adaptive_simpson(f, a, b, absolute_tolerance, max_step_size):
    """
    Integrate f from a to b using Adaptive Simpson's Rule with a maximum error of eps.
    """
    f_values = {x: f(x, absolute_tolerance) for x in (a, b, (a + b) / 2)}
    return _asr(f_values, f, a, b, absolute_tolerance, max_step_size)


def _example_use_simpson():
    import math

    a, b = 0.0, 1.0

    def fun(x, eps):
        print(f"Function call {x:.2f}")
        return math.sin(x)

    sin_integral = integrate_with_adaptive_simpson(fun, a, b, 1e-4)
    print(
        "Simpson's integration of sine from {} to {} = {}\n".format(a, b, sin_integral)
    )
