//In ra một đường thẳng bằng dấu sao
#include <stdio.h>
#include <stdlib.h>
#include <conio.h>

void line() {
    int i;
    for(i = 0; i < 19; i++)
    printf("*");
    printf("\n");
}
void main() {
    system("cls");
    line();
    printf("* Minh hoa ve ham * ");
line();
getch();
}