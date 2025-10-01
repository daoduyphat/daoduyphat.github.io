#include <stdio.h>
#include <stdlib.h>

int main() {
    system("cls");
    int a, b, c;
    int max, min;

    // Nhap 3 so
    printf("Nhap so thu nhat: ");
    scanf("%d", &a);

    printf("Nhap so thu hai: ");
    scanf("%d", &b);

    printf("Nhap so thu ba: ");
    scanf("%d", &c);

    // Tim so lon nhat
    max = a;
    if (b > max) max = b;
    if (c > max) max = c;

    // TTim so nho nhat
    min = a;
    if (b < min) min = b;
    if (c < min) min = c;

    // In ket qua
    printf("So lon nhat la: %d\n", max);
    printf("So nho nhat la: %d\n", min);

    return 0;
}
