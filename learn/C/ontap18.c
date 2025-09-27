//Tính tổng x^1 + x^2 + x^3 + ... + x^n
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("cls");
    //Viết chương trình C: In ra tổng  x^1 + x^2 + x^3 + … + x^n
    //Với hai số nguyên dương (n, x ) được nhập từ bàn phím.
    int x;
    int n;
    int luythua = 1;
    int tong = 0;
    printf("Tinh tong = x^1 + x^2 + .. + x^n\n");
    printf("Nhap x: ");
    scanf("%d",&x);
    printf("Nhap n: ");
    scanf("%d",&n);
    //Nhap gia tri x, n
    for(int i = 1; i <= n; i++){
        luythua = luythua * x;
        tong += luythua;
    }
    printf("%d",tong);
    return 0;
}
