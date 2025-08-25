//In ra 2 mu n
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

int power(int, int);
int power(int ix, int in) {
    int i, ip = 1;
    for (i = 1; i <= in; i++) {
        ip *= ix;
    }
    return ip;
}
int main() {
    system("cls");
    printf("2 mu 2 = %d\n", power(2, 2));
    printf("2 mu 3 = %d\n", power(2, 3));
    getch();
    return 0;
}
