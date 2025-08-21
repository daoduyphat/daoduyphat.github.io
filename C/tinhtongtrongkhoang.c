#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("cls");
    printf("Nhap so dau tien: ");
    int a;
    scanf("%d", &a);
    printf("Nhap so thu 2: ");
    int b;
    scanf("%d", &b);
if ( a > b) {
    printf("Phep tinh khong hop le. Vui long nhap so thu 2 lon hon so thu nhat.\n");
    return 1;
}
int sum = 0;
for (int i = a; i <=b; i++) {
    if (i%2 == 0) {
        sum += i;
    
    }
}
printf("Tong cac so chan trong khoang tu %d den %d la: %d\n", a, b, sum);
return 0;
}

