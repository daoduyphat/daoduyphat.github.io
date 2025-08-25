/*
    Exercise 2: Quản lý sách trong hiệu sách

    ✅ Yêu cầu:
    - Lưu thông tin sách bằng struct book gồm:
        + author_name: tên tác giả
        + book_title: tên sách
        + instock_quantity: số lượng còn trong kho
        + sold_quantity: số lượng đã bán

    ✅ Tính năng chương trình:
    1. Nhập N sách
    2. Hiển thị thông tin tất cả sách
    3. Tìm kiếm sách theo tên tác giả
    4. Thêm một cuốn sách mới
    5. Tính tổng số sách còn trong kho: instock - sold
    6. Sắp xếp sách theo số thứ tự tăng dần của instock_quantity
    0. Thoát

    📌 Mỗi chức năng được viết thành hàm riêng. Dữ liệu được lưu trong mảng.
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
    printf("Enter number of books: ");
    scanf("%d", &n);
    for (int i = 0; i < n; i++) {
        if (count >= MAX_BOOKS) {
            printf("Bookstore is full!\n");
            break;
        }
        printf("Book %d:\n", count + 1);
        printf("  Author name: ");
        scanf(" %[^\n]%*c", books[count].author_name);
        printf("  Book title: ");
        scanf(" %[^\n]%*c", books[count].book_title);
        printf("  In-stock quantity: ");
        scanf("%d", &books[count].instock_quantity);
        printf("  Sold quantity: ");
        scanf("%d", &books[count].sold_quantity);
        count++;
    }
}

void printBooks(struct book books[]) {
    if (count == 0) {
        printf("No books to display.\n");
        return;
    }
    printf("\nList of books:\n");
    for (int i = 0; i < count; i++) {
        printf("Book %d:\n", i + 1);
        printf("  Author name: %s\n", books[i].author_name);
        printf("  Book title: %s\n", books[i].book_title);
        printf("  In-stock: %d\n", books[i].instock_quantity);
        printf("  Sold: %d\n\n", books[i].sold_quantity);
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
            printf("  Author: %s\n", books[i].author_name);
            printf("  Title: %s\n", books[i].book_title);
            printf("  In-stock: %d\n", books[i].instock_quantity);
            printf("  Sold: %d\n\n", books[i].sold_quantity);
            found = 1;
        }
    }
    if (!found) {
        printf("No books found by author: %s\n", author);
    }
}

void insertBook(struct book books[]) {
    if (count >= MAX_BOOKS) {
        printf("Cannot insert more books, store is full.\n");
        return;
    }
    printf("Enter new book info:\n");
    printf("  Author name: ");
    scanf(" %[^\n]%*c", books[count].author_name);
    printf("  Book title: ");
    scanf(" %[^\n]%*c", books[count].book_title);
    printf("  In-stock quantity: ");
    scanf("%d", &books[count].instock_quantity);
    printf("  Sold quantity: ");
    scanf("%d", &books[count].sold_quantity);
    count++;
    printf("Book inserted.\n");
}

void calculateTotalBooks(struct book books[]) {
    int total = 0;
    for (int i = 0; i < count; i++) {
        total += (books[i].instock_quantity - books[i].sold_quantity);
    }
    printf("Total books in store (available): %d\n", total);
}

void sortBooksByStock(struct book books[]) {
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (books[i].instock_quantity > books[j].instock_quantity) {
                struct book temp = books[i];
                books[i] = books[j];
                books[j] = temp;
            }
        }
    }
    printf("Books sorted by in-stock quantity.\n");
}

int main() {
    system("cls");
    struct book books[MAX_BOOKS];
    int choice;

    do {
        printf("\n====== BOOKSTORE MENU ======\n");
        printf("1. Enter N books\n");
        printf("2. Print all books\n");
        printf("3. Search books by author\n");
        printf("4. Insert a new book\n");
        printf("5. Calculate total books in store\n");
        printf("6. Sort books by in-stock quantity\n");
        printf("0. Exit\n");
        printf("Choose your option: ");
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
                calculateTotalBooks(books);
                break;
            case 6:
                sortBooksByStock(books);
                break;
            case 0:
                printf("Goodbye!\n");
                break;
            default:
                printf("Invalid choice! Try again.\n");
        }
    } while (choice != 0);

    return 0;
}
