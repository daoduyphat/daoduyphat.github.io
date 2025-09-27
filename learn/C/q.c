#include <stdio.h>
#include <stdlib.h>

int power(int, int);

int power(int ix, int in) {
    int i, ip = 1;
    for(i = 1; i <= in; i++) {
        ip *= ix; // Tinh luy thua
    }
    return ip; // Tra ve ket qua
}

int main(void) {
    system("cls");
    printf("2 mu 2 = %d\n", power(2, 2)); // Goi ham power
    printf("2 mu 3 = %d\n", power(2, 3)); // Goi ham power
    return 0;
}