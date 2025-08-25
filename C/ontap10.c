//Xêp loại học sinh
#include <stdio.h>
#include <stdlib.h>

int main() {
    system("cls");
    char xeploai = 'B';
switch (xeploai) {
    case 'A':
        printf("Xep loai A\n");
        break;
    case 'B':
    case 'C':
        printf("Kha\n");
        break;
    case 'D':
        printf("Trung binh\n");
        break;
    case 'F':
        printf("Kem\n");
        break;
    default:
        printf("Dup\n");
}
printf("Ban xep loai %c\n", xeploai);
return 0;
}