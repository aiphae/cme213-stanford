#include <thread>
#include <future>
#include <chrono>
#include <iostream>
#include <cassert>
#include <vector>

void f1() {
    std::cout << "f1() called\n";
}

void f2(int m) {
    // Optional: make thread wait a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::cout << "f2() called with m = " << m << std::endl;
}

void f3(int &k) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::cout << "f3() called; k is passed by reference; k = " << k << std::endl;
    k += 3;
}

void accumulate(std::vector<int>::iterator first, std::vector<int>::iterator last, std::promise<int> accumulate_promise) {
    int sum = 0;
    auto it = first;
    for (; it != last; ++it) {
        sum += *it;
    }
    accumulate_promise.set_value(sum); // Notify future
}

void get_max(std::vector<int>::iterator first, std::vector<int>::iterator last, std::promise<int> max_promise) {
    int max = *first;
    auto it = first;
    for (; it != last; ++it) {
        max = (*it > max ? *it : max);
    }
    max_promise.set_value(max);
}

int main(void) {
    // Demonstrate using thread constructor
    std::thread t1(f1);

    int m = 5;
    // With an argument
    std::thread t2(f2, m);

    int k = 7;
    // With a reference
    std::thread t3(f3, std::ref(k)); // Use ref to pass a reference

    // Wait for all threads to finish */
    t1.join();
    t2.join();
    t3.join();

    std::cout << "k is now equal to " << k << std::endl;
    assert(k == 10);

    std::thread t4([&k, m]() { k += m; });
    t4.join();

    std::cout << "k is now equal to " << k << std::endl;
    assert(k == 15);

    // Demonstrate using std::promise<int> to return a value
    std::vector<int> vec_1 = {1, 2, 3, 4, 5, 6};
    std::promise<int> accumulate_promise; // Will store the int
    // Used to retrieve the value asynchronously, at a later time
    std::future<int> accumulate_future = accumulate_promise.get_future();

    // move() will "move" the resources allocated for accumulate_promise
    std::thread t5(accumulate, vec_1.begin(), vec_1.end(), std::move(accumulate_promise));

    // future::get() waits until the future has a valid result and retrieves it
    std::cout << "result of accumulate_future [21 expected] = " << accumulate_future.get() << '\n';
    t5.join(); // Wait for thread completion

    std::vector<int> vec_2 = {1, -2, 4, -10, 5, 4};
    std::promise<int> max_promise;
    std::future<int> max_future = max_promise.get_future();

    std::thread t6(get_max, vec_2.begin(), vec_2.end(), std::move(max_promise));

    int max_result = max_future.get();
    std::cout << "result of max_future [5 expected] = " << max_result << '\n';
    assert(max_result == 5);

    t6.join(); // Wait for thread completion

    return 0;
}