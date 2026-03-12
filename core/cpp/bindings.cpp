#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "black_scholes.h"
#include "iv_solver.h"
#include "svi_calibrator.h"
#include <vector>
#include <cmath>

namespace py = pybind11;

py::array_t<double> bs_price_vec(
    double S, py::array_t<double> K_arr, double r, double q,
    py::array_t<double> v_arr, py::array_t<double> T_arr,
    py::array_t<int> cp_arr)
{
    auto K = K_arr.unchecked<1>(); auto v = v_arr.unchecked<1>();
    auto T = T_arr.unchecked<1>(); auto cp = cp_arr.unchecked<1>();
    size_t n = K.shape(0);
    py::array_t<double> result(n);
    auto out = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; ++i)
        out(i) = bs_price(S, K(i), r, q, v(i), T(i), cp(i));
    return result;
}

py::array_t<double> implied_vol_vec(
    double S, py::array_t<double> K_arr, double r, double q,
    py::array_t<double> price_arr, py::array_t<double> T_arr,
    py::array_t<int> cp_arr)
{
    auto K = K_arr.unchecked<1>(); auto price = price_arr.unchecked<1>();
    auto T = T_arr.unchecked<1>(); auto cp = cp_arr.unchecked<1>();
    size_t n = K.shape(0);
    py::array_t<double> result(n);
    auto out = result.mutable_unchecked<1>();
    for (size_t i = 0; i < n; ++i)
        out(i) = implied_vol(S, K(i), r, q, price(i), T(i), cp(i));
    return result;
}

PYBIND11_MODULE(vol_core, m) {
    m.doc() = "Vol Arb Engine — C++ core";

    py::class_<Greeks>(m, "Greeks")
        .def_readonly("delta", &Greeks::delta)
        .def_readonly("gamma", &Greeks::gamma)
        .def_readonly("vega",  &Greeks::vega)
        .def_readonly("theta", &Greeks::theta)
        .def_readonly("rho",   &Greeks::rho);

    m.def("bs_price",  &bs_price,  "BS price",
          py::arg("S"),py::arg("K"),py::arg("r"),py::arg("q"),
          py::arg("v"),py::arg("T"),py::arg("call_put"));
    m.def("bs_greeks", &bs_greeks, "BS Greeks",
          py::arg("S"),py::arg("K"),py::arg("r"),py::arg("q"),
          py::arg("v"),py::arg("T"),py::arg("call_put"));
    m.def("implied_vol", &implied_vol, "IV solver",
          py::arg("S"),py::arg("K"),py::arg("r"),py::arg("q"),
          py::arg("market_price"),py::arg("T"),py::arg("call_put"));
    m.def("bs_price_vec",    &bs_price_vec,    "Vectorized BS price",
          py::arg("S"),py::arg("K"),py::arg("r"),py::arg("q"),
          py::arg("v"),py::arg("T"),py::arg("call_put"));
    m.def("implied_vol_vec", &implied_vol_vec, "Vectorized IV solver",
          py::arg("S"),py::arg("K"),py::arg("r"),py::arg("q"),
          py::arg("market_price"),py::arg("T"),py::arg("call_put"));

    // ── SVI ──────────────────────────────────────────────────────
    py::class_<SVIParams>(m, "SVIParams")
        .def(py::init<>())
        .def_readwrite("a",     &SVIParams::a)
        .def_readwrite("b",     &SVIParams::b)
        .def_readwrite("rho",   &SVIParams::rho)
        .def_readwrite("m",     &SVIParams::m)
        .def_readwrite("sigma", &SVIParams::sigma)
        .def("__repr__", [](const SVIParams& p) {
            return "SVIParams(a=" + std::to_string(p.a)
                 + ", b="    + std::to_string(p.b)
                 + ", rho="  + std::to_string(p.rho)
                 + ", m="    + std::to_string(p.m)
                 + ", sigma="+ std::to_string(p.sigma) + ")";
        });

    py::class_<SVIResult>(m, "SVIResult")
        .def_readonly("params",     &SVIResult::params)
        .def_readonly("rmse",       &SVIResult::rmse)
        .def_readonly("converged",  &SVIResult::converged)
        .def_readonly("iterations", &SVIResult::iterations);

    m.def("svi_w",   &svi_w,   "SVI total variance w(k)",
          py::arg("k"), py::arg("params"));
    m.def("svi_vol", &svi_vol, "SVI implied vol",
          py::arg("k"), py::arg("T"), py::arg("params"));
    m.def("calibrate_svi", &calibrate_svi,
          "Calibrate SVI via Levenberg-Marquardt",
          py::arg("log_moneyness"), py::arg("market_ivs"),
          py::arg("T"), py::arg("init") = SVIParams{});
    m.def("svi_vol_vec", [](py::array_t<double> k_arr, double T, const SVIParams& p) {
        auto k = k_arr.unchecked<1>();
        size_t n = k.shape(0);
        py::array_t<double> out(n);
        auto o = out.mutable_unchecked<1>();
        for (size_t i = 0; i < n; ++i) o(i) = svi_vol(k(i), T, p);
        return out;
    }, "Vectorized SVI vol", py::arg("k"), py::arg("T"), py::arg("params"));
}
