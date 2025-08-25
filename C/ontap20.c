//Trung bình cộng của n số nguyên
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

int main(void) {
    int ia[50], i, in, isum = 0;
    system("cls");
    printf("Nhap vao gia tri n: ");
    scanf("%d", &in);
    for(i = 0; i < in; i++) {
        printf("Nhap vao phan tu thu %d: ", i + 1);
        scanf("%d", &ia[i]);
    }
    for(i = 0; i < in; i++)
    isum += ia[i];
    printf("Trung binh cong: %.2f", (float)isum / in);
    getch();
}