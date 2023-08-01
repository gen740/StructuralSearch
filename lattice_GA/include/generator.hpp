#pragma once
#include <coroutine>
#include <exception>
#include <memory>
#include <type_traits>

namespace Lattice {

template <typename T>
struct is_unique_ptr : std::false_type {};

template <typename T>
struct is_unique_ptr<std::unique_ptr<T>> : std::true_type {};

/*!
 * 最低限の Generator の実装アルゴリズムとかで使うと良い
 * C++23 でstd に追加されたらそれに変えること
 * */
template <class ValType>
struct Generator {
  struct promise_type;
  using handle = std::coroutine_handle<promise_type>;
  struct promise_type {
    ValType current_value;
    static auto get_return_object_on_allocation_failure() {
      return Generator{nullptr};
    }
    auto get_return_object() { return Generator{handle::from_promise(*this)}; }
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    void unhandled_exception() { std::terminate(); }
    void return_void() {}
    // void return_value(ValType value) {current_value = value;}
    auto yield_value(ValType value) {
      if constexpr (is_unique_ptr<ValType>::value) {
        current_value = std::move(value);
      } else {
        current_value = value;
      }
      return std::suspend_always{};
    }
  };

  bool next() {
    if (coro) [[likely]] {
      coro();
      return !coro.done();
    }
    return false;
  }

  ValType value() {
    if constexpr (is_unique_ptr<ValType>::value) {
      return std::move(coro.promise().current_value);
    } else {
      return coro.promise().current_value;
    }
  }

  Generator &operator=(const Generator &) = delete;
  Generator &operator=(Generator &&) = delete;
  Generator(Generator const &) = delete;
  Generator(Generator &&rhs) noexcept : coro(rhs.coro) { rhs.coro = nullptr; }
  ~Generator() {
    if (coro) {
      coro.destroy();
    }
  }

  struct Iter {
    Generator &gen;
    bool operator!=(const Iter &) const { return gen.next(); }
    void operator++() {}
    const ValType operator*() {
      if constexpr (is_unique_ptr<ValType>::value) {
        return std::move(gen.value());
      } else {
        return gen.value();
      }
    }
  };
  Iter begin() { return Iter{*this}; }
  Iter end() { return Iter{*this}; }

 private:
  explicit Generator(handle h) : coro(h) {}
  handle coro;
};

}  // namespace Lattice
