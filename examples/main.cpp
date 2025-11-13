#include <wolf.h> 

using namespace wolf;

int main() {
    wolf::Tensor a;
    Sequential b(Linear(5, 3), ReLU(), Linear(3, 6));
    Tensor y = b.pred(Tensor({0.5, 1, 2, 3, 4}, 5, 1));
    Tensor t({4, 3, 6, 0, 4, 2}, 6, 1);
    Tensor l = loss(y, t);
    b.backward(l);
    // b.step();
    // for (size_t i = 0; i < c.nrows() * c.ncols(); i++) {
    //     std::cout << c(i) << ' ';
    // }
    
}