//In ra thông tin của nhân viên
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
struct  employee {
    int id;
    char name[50];
    float salary;
};
int main(){
    system("cls");
    struct employee e1, e2;
    e1.id = 71;
    strcpy(e1.name, "Nguyen Van A");
    e1.salary = 1000;
    e2.id = 37;
    strcpy(e2.name, "Tran Van B");
    e2.salary = 500;
    printf("employee 1 id:%d\n", e1.id);
    printf("employee 1 name:%s\n", e1.name);
    printf("employee 1 salary:%f\n", e1.salary);
    printf("employee 2 id:%d\n", e2.id);
    printf("employee 2 name:%s\n", e2.name);
    printf("employee 2 salary:%f\n", e2.salary);
    return 0;
}