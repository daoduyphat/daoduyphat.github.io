#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("cls");
    int a,b;
    printf("Nhap a = ");
    scanf("%d", &a);
    printf("Nhap b = ");
    scanf("%d", &b);
    printf("Tong cua 2 so %d va %d la %d\n", a, b, a+b);
    return 0;
}