//Tính tổng các số chẵn trong khoảng từ a đến b
#include <stdio.h>
#include <stdlib.h>

int main() {
    system("cls");
    int a, b;
    int sum = 0;
    printf("Nhap a = ");
    scanf("%d", &a);
    printf("Nhap b = ");
    scanf("%d", &b);
    if (a > b) {
        printf("Khong hop le!");
        return 0;
    }
    for (int i = a; i <= b; i++) {
        if (i % 2 == 0) {
            sum += i;
        }
    }
    printf("Tong cac so chan trong khoang [%d, %d] la: %d", a, b, sum);
    return 0;
}