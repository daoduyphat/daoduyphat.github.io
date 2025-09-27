//Viết chương trình nhập vào một số nguyên dương có 2 chữ số và in cách đọc ra màn hình
#include <stdio.h>
#include <stdlib.h>

int main() {
    system("cls");
    int n;
    printf("Nhap so duong co 2 chu so: ");
    scanf("%d", &n);

    if (n < 10 || n > 99) {
        printf("So nhap vao khong phai la so co 2 chu so.\n");
        return 1;
    }

    int chuc = n / 10;
    int donvi = n % 10;

    printf("Cach doc: ");

    // Đọc hàng chục
    switch (chuc) {
        case 1: printf("Muoi "); break;
        case 2: printf("Hai muoi "); break;
        case 3: printf("Ba muoi "); break;
        case 4: printf("Bon muoi "); break;
        case 5: printf("Nam muoi "); break;
        case 6: printf("Sau muoi "); break;
        case 7: printf("Bay muoi "); break;
        case 8: printf("Tam muoi "); break;
        case 9: printf("Chin muoi "); break;
    }

    if (donvi == 0) {
    } else if (donvi == 1 && chuc != 1) {
        printf("mot"); // "mốt"
    } else if (donvi == 5 && chuc != 0) {
        printf("lam"); // "lăm"
    } else if (donvi == 4 && chuc != 0) {
        printf("tu");  // "tư"
    } else {
        switch (donvi) {
            case 1: printf("mot"); break;
            case 2: printf("hai"); break;
            case 3: printf("ba"); break;
            case 4: printf("bon"); break;
            case 5: printf("nam"); break;
            case 6: printf("sau"); break;
            case 7: printf("bay"); break;
            case 8: printf("tam"); break;
            case 9: printf("chin"); break;
        }
    }
    return 0;
}
