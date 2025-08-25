// Số chẵn số lẻ
#include <stdio.h>
#include <stdlib.h>

void main() {
    system("cls");
    int num;
    printf("Nhap so num = ");
    scanf("%d", &num);
    if(num%2 == 0){
        printf("num la so chan\n");
    } else {
        printf("num la so le");
    }
}