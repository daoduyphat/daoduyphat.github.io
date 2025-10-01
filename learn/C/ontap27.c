//Sử dụng cấu trúc để lưu trữ thông tin của nhân viên
// Chương trình này cho phép người dùng nhập thông tin của nhiều nhân viên,
#include <stdio.h>
#include <stdlib.h>
struct employee
{
    int rollno;
    int salary;
    //…
};
void accept(struct employee list[80], int s)
{
    int i;
    for (i = 0; i < s; i++)
    {
        printf("\nEnter data for Record #%d", i + 1);
        printf("\nEnter rollno : ");
        scanf("%d", &list[i].rollno);
        fflush(stdin);
        printf("Enter salary : ");
        scanf("%d", &list[i].salary);
    }
}
\
void display(struct employee list[80], int s)
{
    int i;
    printf("\n\nRollno\tName\tsalary\n");
    for (i = 0; i < s; i++)
    {
        printf("%d\t\t%d\n", list[i].rollno, list[i].salary);
    }
}
void search(struct employee list[80], int s, int number)
{
    int i;
    for (i = 0; i < s; i++)
    {
        if (list[i].rollno == number)
        {
            printf("Rollno : %d\nsalary : %d\n", list[i].rollno, list[i].salary);
            return ;
        }
    }
    printf("Record not Found\n");
}
int main()
{
    system("cls");
    struct employee data[20];
    int n, choice, rollno;
    printf("Number of records you want to enter? : ");
    scanf("%d", &n);
    accept(data, n);
    do
    {
        printf("\nResult Menu :\n");
        printf("Press 1 to display all records.\n");
        printf("Press 2 to search a record.\n");
        printf("Press 0 to exit\n");
        printf("\nEnter choice(0-2) : ");
        scanf("%d", &choice);
        switch (choice)
        {
            case 1:
                display(data, n);
                break;
            case 2:
                printf("Enter roll number to search : ");
                scanf("%d", &rollno);
                search(data, n, rollno);
                break;
        }
    }
    while (choice != 0);
    return 0;
}