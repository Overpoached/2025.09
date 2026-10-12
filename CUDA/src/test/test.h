#pragma once

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace test {
	// Moves every TEST_CASE into testCases. Idempotent; RunAllTests calls it for you.
	void InitTests();
	// Returns the number of failed tests, so the caller can turn it into an exit code.
	int RunAllTests();

	namespace detail {
		// Thrown by the EXPECT_* macros; the runner catches it and records the message.
		struct Failure : std::runtime_error {
			using std::runtime_error::runtime_error;
		};

		// One static node per TEST_CASE, chained into an intrusive list at static-init time.
		// That is a single pointer write per test: no allocation, nothing in testCases.
		// The list is only walked (and testCases filled) when InitTests() is called.
		struct Registrar {
			const char* name;
			void (*fn)();
			Registrar* next;
			Registrar(const char* name, void (*fn)());
		};

		inline std::string Where(const char* file, int line) {
			return std::string(file) + "(" + std::to_string(line) + "): ";
		}
	}
}

// TEST_CASE(Name) { ...body... }
#define TEST_CASE(name)                                                              \
	static void TestCase_##name();                                                   \
	static ::test::detail::Registrar registrar_##name(#name, &TestCase_##name);      \
	static void TestCase_##name()

// Both macros abort the current test on the first failure (later checks would be noise).
#define EXPECT_EQ(actual, expected)                                                  \
	do {                                                                             \
		const auto& a_ = (actual);                                                   \
		const auto& e_ = (expected);                                                 \
		if (!(a_ == e_)) {                                                           \
			std::ostringstream os_;                                                  \
			os_ << ::test::detail::Where(__FILE__, __LINE__)                         \
			    << "EXPECT_EQ(" #actual ", " #expected ") got " << a_                 \
			    << ", expected " << e_;                                              \
			throw ::test::detail::Failure(os_.str());                                \
		}                                                                            \
	} while (0)

// Floats: |actual - expected| <= tolerance.
#define EXPECT_NEAR(actual, expected, tolerance)                                     \
	do {                                                                             \
		const double a_ = static_cast<double>(actual);                               \
		const double e_ = static_cast<double>(expected);                             \
		const double t_ = static_cast<double>(tolerance);                            \
		if (!(std::fabs(a_ - e_) <= t_)) { /* written this way so NaN fails too */   \
			std::ostringstream os_;                                                  \
			os_.precision(9);                                                        \
			os_ << ::test::detail::Where(__FILE__, __LINE__)                         \
			    << "EXPECT_NEAR(" #actual ", " #expected ") got " << a_               \
			    << ", expected " << e_ << " (diff " << std::fabs(a_ - e_)            \
			    << " > tol " << t_ << ")";                                           \
			throw ::test::detail::Failure(os_.str());                                \
		}                                                                            \
	} while (0)
