#include <stdio.h>
#include <stdlib.h>

// Định nghĩa hàm line
void line() {
    system("cls"); 
    printf("--------------------\n");
}

int main() {
    line();
    printf("*Minh hoa ve ham*\n");
    line();
    getchar(); // thay cho getch()
    return 0;
}
