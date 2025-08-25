/*
    Chương trình quản lý sách trong cửa hàng.

    ✅ Chức năng chính:
    1. Nhập nhiều sách (N sách cùng lúc).
    2. Hiển thị thông tin tất cả sách.
    3. Tìm và hiển thị sách theo tên tác giả.
    4. Thêm một cuốn sách mới.
    5. Sắp xếp danh sách sách theo số lượng còn trong kho (tăng dần).

    📌 Mỗi cuốn sách bao gồm:
       - Tên tác giả
       - Tên sách
       - Số lượng còn trong kho
       - Số lượng đã bán

    ⚠️ Giới hạn tối đa: 100 cuốn sách.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define MAX_BOOKS 100

struct book {
    char author_name[30];
    char book_title[30];
    int instock_quantity;
    int sold_quantity;
};

int count = 0;

void enterBooks(struct book books[]) {
    int n;
    printf("Enter the number of books: ");
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        printf("Enter details for book %d:\n", count + 1);
        printf("Author Name: ");
        scanf(" %[^\n]%*c", books[count].author_name);
        printf("Book Title: ");
        scanf(" %[^\n]%*c", books[count].book_title);
        printf("In-stock Quantity: ");
        scanf("%d", &books[count].instock_quantity);
        printf("Sold Quantity: ");
        scanf("%d", &books[count].sold_quantity);
        count++;
    }
}

void printBooks(struct book books[]) {
    if (count == 0) {
        printf("No books in the store.\n");
        return;
    }
    for (int i = 0; i < count; i++) {
        printf("Book %d:\n", i + 1);
        printf("Author Name: %s\n", books[i].author_name);
        printf("Book Title: %s\n", books[i].book_title);
        printf("In-stock Quantity: %d\n", books[i].instock_quantity);
        printf("Sold Quantity: %d\n", books[i].sold_quantity);
        printf("\n");
    }
}

void searchByAuthor(struct book books[]) {
    char author[30];
    printf("Enter author name to search: ");
    scanf(" %[^\n]%*c", author);
    int found = 0;
    for (int i = 0; i < count; i++) {
        if (strcmp(books[i].author_name, author) == 0) {
            printf("Book %d:\n", i + 1);
            printf("Author Name: %s\n", books[i].author_name);
            printf("Book Title: %s\n", books[i].book_title);
            printf("In-stock Quantity: %d\n", books[i].instock_quantity);
            printf("Sold Quantity: %d\n", books[i].sold_quantity);
            printf("\n");
            found = 1;
        }
    }
    if (!found) {
        printf("No books found for author: %s\n", author);
    }
}

void insertBook(struct book books[]) {
    if (count >= MAX_BOOKS) {
        printf("Store is full, cannot add more books.\n");
        return;
    }
    printf("Enter details for the new book:\n");
    printf("Author Name: ");
    scanf(" %[^\n]%*c", books[count].author_name);
    printf("Book Title: ");
    scanf(" %[^\n]%*c", books[count].book_title);
    printf("In-stock Quantity: ");
    scanf("%d", &books[count].instock_quantity);
    printf("Sold Quantity: ");
    scanf("%d", &books[count].sold_quantity);
    count++;
}

void sortBooksByStock(struct book books[]) {
    struct book temp;
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (books[i].instock_quantity > books[j].instock_quantity) {
                temp = books[i];
                books[i] = books[j];
                books[j] = temp;
            }
        }
    }
    printf("Books have been sorted by in-stock quantity.\n");
}

int main() {
    system("cls");
    struct book books[MAX_BOOKS];
    int choice;

    do {
        printf("\n============================\n");
        printf("   BOOK STORE MANAGEMENT\n");
        printf("============================\n");
        printf("1. Enter N books\n");
        printf("2. Print/show all books\n");
        printf("3. Search books by author\n");
        printf("4. Insert a new book\n");
        printf("5. Sort books by in-stock quantity\n");
        printf("0. Exit\n");
        printf("Your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                enterBooks(books);
                break;
            case 2:
                printBooks(books);
                break;
            case 3:
                searchByAuthor(books);
                break;
            case 4:
                insertBook(books);
                break;
            case 5:
                sortBooksByStock(books);
                break;
            case 0:
                printf("Exiting program...\n");
                break;
            default:
                printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 0);

    return 0;
}
