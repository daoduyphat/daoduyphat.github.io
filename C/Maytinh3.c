// Cũng là máy tính nhưng cách khác
#include <stdio.h>      // Thư viện cho nhập xuất chuẩn (printf, scanf)
#include <stdlib.h>     // Thư viện cho các hàm tiện ích, như system()

int main(void) {
    system("cls");      // Xóa màn hình console (chỉ hoạt động trên Windows)
    char op;            // Biến lưu phép toán (+, -, *, /)
    double first, second;   // Hai biến lưu hai số thực nhập vào
    printf("Nhap so thu nhat: ");      // Hiển thị yêu cầu nhập số thứ nhất
    scanf("%lf", &first);              // Nhập số thực đầu tiên từ bàn phím
    getchar();                         // Đọc ký tự '\n' còn lại trong bộ đệm
    printf("Nhap phep toan (+, -, *, /): "); // Hiển thị yêu cầu nhập phép toán
    scanf("%c", &op);                  // Nhập ký tự phép toán từ bàn phím
    getchar();                         // Đọc ký tự '\n' còn lại trong bộ đệm
    printf("Nhap so thu hai: ");       // Hiển thị yêu cầu nhập số thứ hai
    scanf("%lf", &second);             // Nhập số thực thứ hai từ bàn phím
    getchar();                         // Đọc ký tự '\n' còn lại trong bộ đệm
    switch (op) {                      // Kiểm tra phép toán vừa nhập
        case '+':                      // Nếu là phép cộng
            printf("%.1lf + %.1lf = %.1lf", first, second, first + second); // Cộng hai số và in kết quả
            break;
        case '-':                      // Nếu là phép trừ
            printf("%.1lf - %.1lf = %.1lf", first, second, first - second); // Trừ hai số và in kết quả
            break;
        case '*':                      // Nếu là phép nhân
            printf("%.1lf * %.1lf = %.1lf", first, second, first * second); // Nhân hai số và in kết quả
            break;
        case '/':                      // Nếu là phép chia
            printf("%.1lf / %.1lf = %.1lf", first, second, first / second); // Chia hai số và in kết quả
            break;
        default:                       // Nếu nhập sai phép toán
            printf("Phep toan khong hop le"); // Báo lỗi phép toán không hợp lệ
    }
    return 0;   // Kết thúc chương trình
}