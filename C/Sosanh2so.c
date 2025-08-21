//Dùng để so sánh hai số nguyên
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int a, b;
    system("cls");
    printf("Nhap vao so a: ");
    scanf("%d", &a);
    printf("Nhap vao so b: ");
    scanf("%d", &b);
    if (a == b) 
        printf("a bang b \n");
    else if (a > b) 
        printf("a lon hon b \n");
    else 
        printf("a nho hon b \n");
    system("pause");
    return 0;
}