//ss lớn nhỏ
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int a, b;
    system("cls");
    printf("Nhap a = ");
    scanf("%d", &a);
    printf("Nhap b = ");
    scanf("%d", &b);
    if (a > b) {
        printf("a lon hon b\n");
    } else if (a < b) {
        printf("a nho hon b\n");
    } else {
        printf("a bang b\n");
    }
    return 0;
}