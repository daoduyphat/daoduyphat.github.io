#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    int sum = a + b;
    return sum;
}   
int main() {
    int a = 3, b =4;
    printf("\n%d + %d = %d\n", a, b, add(a, b));
    return 0;
}