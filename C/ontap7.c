//In ra chuỗi vừa nhập
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    system("cls");
    char name[50];
    printf("Nhap chuoi: ");
    fgets(name, sizeof(name), stdin);
    printf("Name: ");
    puts(name);
    return 0;


}