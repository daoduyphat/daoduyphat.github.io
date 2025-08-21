#include <stdio.h>
#include <stdlib.h>

void getInput() {
    system("cls");
    char c;
    printf("Nhap du lieu: ");
    scanf("%c", &c);

    if(c == 'q' || c == 'Q') {
        printf("The input is Q, quitting now...\n");
        exit(0);
    } else {
        printf("Ban da nhap: %c\n", c);
    }
}

int main() {
    getInput();
    return 0;
}   