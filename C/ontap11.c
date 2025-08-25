//Tính toán với 2 số thực nhập vào
#include <stdio.h>
#include <stdlib.h>

int main() {
    system("cls");
    double a;
    printf("Nhap so a: ");
    scanf("%lf", &a);
    char op;
    printf("Nhap phep toan (+, -, *, /): ");
    scanf(" %c", &op);
    double b;
    printf("Nhap so b: ");
    scanf("%lf", &b);
    switch (op) {
    case '+':
        printf("Tong cua %lf va %lf la: %lf\n", a, b, a + b);
        break;
    case '-':
        printf("Hieu cua %lf va %lf la: %lf\n", a, b, a - b);
        break;
    case '*':
        printf("Tich cua %lf va %lf la: %lf\n", a, b, a * b);
        break;
    case '/':
        if (b != 0) {
            printf("Thuong cua %lf va %lf la: %lf\n", a, b, a / b);
        } else {
            printf("Khong the chia cho 0\n");
        }
        break;
    
    default:
        printf("Phep toan khong hop le\n");
        break;
    }
}