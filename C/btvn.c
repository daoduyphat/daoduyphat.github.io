#include <stdio.h>
#include <stdlib.h>

#define MAX 100


struct nhanvien {
    int manv;     
    float luong;   
};


void inputEmployees(struct nhanvien emp[], int n) {
    for (int i = 0; i < n; i++) {
        printf("\nEnter data for Record #%d\n", i + 1);
        printf("Enter manv: ");
        scanf("%d", &emp[i].manv);
        printf("Enter luong: ");
        scanf("%f", &emp[i].luong);
    }
}


void displayEmployees(struct nhanvien emp[], int n) {
    printf("\n%-10s %-10s\n", "Manv", "Luong");
    for (int i = 0; i < n; i++) {
        printf("%-10d %-10.2f\n", emp[i].manv, emp[i].luong);
    }
}


void searchEmployee(struct nhanvien emp[], int n) {
    int key;
    printf("Enter manv to search: ");
    scanf("%d", &key);
    int found = 0;

    for (int i = 0; i < n; i++) {
        if (emp[i].manv == key) {
            printf("Found!\nManv: %d\nLuong: %.2f\n", emp[i].manv, emp[i].luong);
            found = 1;
            break;
        }
    }

    if (!found) {
        printf("Record not found.\n");
    }
}

int main() {
    system("cls");
    struct nhanvien emp[MAX];
    int n;

    printf("Number of records you want to enter: ");
    scanf("%d", &n);


    if (n < 1 || n > MAX) {
        printf("Invalid number of records. Please enter a number between 1 and %d.\n", MAX);
        return 1;
    }

    inputEmployees(emp, n);

    int choice;

    do {
        
        printf("\nResult Menu:\n");
        printf("Press 1 to display all records.\n");
        printf("Press 2 to search a record.\n");
        printf("Press 0 to exit\n");

        printf("Enter choice (0-2): ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                displayEmployees(emp, n);
                break;
            case 2:
                searchEmployee(emp, n);
                break;
            case 0:
                printf("Exiting program.\n");
                break;
            default:
                printf("Invalid choice! Please try again.\n");
        }

    } while (choice != 0);

    return 0;
}
