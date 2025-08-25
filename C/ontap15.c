//Tính tổng các số từ 1 đến N
#include <stdio.h>
#include <stdlib.h>

int main() {
    system("cls");
    int num, count, sum = 0;
    printf("Nhap so N can tinh: ");
    scanf("%d", &num);
    
    //Vong lap de tinh tong
    for(count = 1; count <= num; count++) {
        sum += count;
    }
    printf("Tong = %d", sum);
    return 0;
}