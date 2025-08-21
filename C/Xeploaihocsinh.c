//Dùng để xếp loại học sinh dựa trên điểm số
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("cls");
    printf("Nhap vao diem cua hoc sinh: ");
    int diem;
    scanf("%d", &diem);
    if (diem > 10 || diem < 0) {
        printf("Diem khong hop le. Vui long nhap lai diem trong khoang tu 0 den 10.\n");
    } else if (diem >= 9) {
        printf("Hoc sinh xuat sac\n");
    } else if (diem >= 8) {
        printf("Hoc sinh gioi\n");
    } else if (diem >= 7) {
        printf("Hoc sinh kha\n");
    } else if (diem >= 6) {
        printf("Hoc sinh tb kha\n");
    } else if (diem >= 5) {
        printf("Hoc sinh trung binh\n");
    } else {
        printf("Hoc sinh yeu\n");
    }
}