#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("Nhap so: ");
    int n;
    scanf("%d", &n);
    if (n % 2 == 0) {
        printf("So %d la so chan\n", n);
    } else {
        printf("So %d la so le\n", n);
    }
}