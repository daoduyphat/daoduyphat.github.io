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
     else
        printf("a khac b \n");
    system("pause");
    return 0;
}