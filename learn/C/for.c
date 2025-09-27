#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int i, j;
    for (i = 1; j <= 5; i++) {
        printf("*");
        for (j = 1; j <= i; j++) {
            printf("*");
        }
        printf("\n");
    }
        return 0;
    }
