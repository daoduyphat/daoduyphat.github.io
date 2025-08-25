//In ra hình tam giác sao
#include <stdio.h>
#include <stdlib.h>
int main() {
    int i,j;
    system("cls");
    for (i = 1; i <=5; i++) {
        for (j = 1; j <= i; j++) {
            printf("*");
        
        }
        printf("\n");
    }
    return 0;
}