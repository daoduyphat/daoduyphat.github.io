//Tính tổng của hai số nguyên
#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    int sum = a + b;
    return sum;
}
int main() {
    system("cls");
    int a = 3, b = 4;
    printf("\n%d + %d = %d", a, b, add(a, b));
    return 0;
}