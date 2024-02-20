def integrate_with_weight(weight, sampler):
    """
    Compute integral of the form

    :math:`\\int_0^\\infty weight(x) \mathbb{sampler(x)} dx`

    Parameters
    ----------
    weight : callable
        weight function. Signature float -> float

    sampler : callable
        probablility source. Signature float x, int n -> list([0,1])

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

    return _integrate_on_interval(weight_transformed, sampler_transformed)


def _integrate_on_interval(weight, sampler):
    """
    Compute integral of the form

    :math:`\\int_0^1 weight(x) \mathbb{sampler(x)} dx`

    Parameters
    ----------
    weight : callable
        weight function. Signature float -> float

    sampler : callable
        probablility source. Signature float x, int n -> list([0,1])

    Returns
    -------
    float
        value of the integral

    """


import math
import collections


def _evaluate_simpsons_rule(f_values, f, a, b):
    """
    Evaluate Simpson's Rule using cached function values.
    """
    m = (a + b) / 2
    fm = f(m)
    f_values[m] = fm
    simp = abs(b - a) / 6 * (f_values[a] + 4 * fm + f_values[b])
    return m, fm, simp


def _adaptive_simpsons_rule(f_values, f, a, b, eps):
    """
    Efficient recursive implementation of adaptive Simpson's rule using cached function values.
    """
    m = (a + b) / 2
    fa, fb, fm = f_values[a], f_values[b], f_values[m]

    left_m, left_fm, left_simp = _evaluate_simpsons_rule(f_values, f, a, m)
    right_m, right_fm, right_simp = _evaluate_simpsons_rule(f_values, f, m, b)
    delta = left_simp + right_simp - (fa + fb) * (b - a) / 2

    if abs(delta) <= eps * 15:
        return left_simp + right_simp + delta / 15
    else:
        return _adaptive_simpsons_rule(
            f_values, f, a, m, eps / 2
        ) + _adaptive_simpsons_rule(f_values, f, m, b, eps / 2)


def integrate_with_asr(f, a, b, eps):
    """
    Integrate f from a to b using Adaptive Simpson's Rule with a maximum error of eps.
    """
    f_values = {x: f(x) for x in (a, b, (a + b) / 2)}
    return _adaptive_simpsons_rule(f_values, f, a, b, eps)


def main():
    a, b = 0.0, 1.0
    def fun(x):
        print(f"Function call {x:.2f}")
        return math.sin(x)
    
    sinx_integration = integrate_with_asr(fun, a, b, 1)
    print(
        "Simpson's integration of sine from {} to {} = {}\n".format(
            a, b, sinx_integration
        )
    )


if __name__ == "__main__":
    main()
