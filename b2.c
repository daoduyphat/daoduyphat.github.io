#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct SinhVien {
    char maSV[10];
    char hoTen[40];
    char nganhHoc[30];
    int namNhapHoc;
};

void nhapSinhVien(struct SinhVien ds[], int *soLuong) {
    printf("Nhap so luong sinh vien: ");
    scanf("%d", soLuong);
    for(int i = 0; i < *soLuong; i++) {
        printf("Nhap sinh vien thu %d:\n", i+1);
        printf("Ma SV: ");
        scanf("%s", ds[i].maSV);
        printf("Ho ten: ");
        getchar();
        fgets(ds[i].hoTen, sizeof(ds[i].hoTen), stdin);
        ds[i].hoTen[strcspn(ds[i].hoTen, "\n")] = '\0';

        printf("Nganh hoc: ");
        fgets(ds[i].nganhHoc, sizeof(ds[i].nganhHoc), stdin);
        ds[i].nganhHoc[strcspn(ds[i].nganhHoc, "\n")] = '\0';

        printf("Nam nhap hoc: ");
        scanf("%d", &ds[i].namNhapHoc);
    }
}

void hienThiSinhVien(struct SinhVien ds[], int soLuong) {
    printf("\nDanh sach sinh vien:\n");
    for(int i = 0; i < soLuong; i++) {
        printf("MaSV: %s, HoTen: %s, Nganh: %s, NamNhapHoc: %d\n",
            ds[i].maSV, ds[i].hoTen, ds[i].nganhHoc, ds[i].namNhapHoc);
    }
}

void timTheoNganh(struct SinhVien ds[], int soLuong) {
    char nganhTim[30];
    printf("Nhap nganh can tim: ");
    getchar();
    fgets(nganhTim, sizeof(nganhTim), stdin);
    nganhTim[strcspn(nganhTim, "\n")] = '\0';

    printf("Sinh vien thuoc nganh %s:\n", nganhTim);
    for(int i = 0; i < soLuong; i++) {
        if(strcmp(ds[i].nganhHoc, nganhTim) == 0) {
            printf("MaSV: %s, HoTen: %s, NamNhapHoc: %d\n",
                ds[i].maSV, ds[i].hoTen, ds[i].namNhapHoc);
        }
    }
}

void themSinhVien(struct SinhVien ds[], int *soLuong) {
    struct SinhVien svMoi;
    int maTrung;
    do {
        maTrung = 0;
        printf("Nhap MaSV moi: ");
        scanf("%s", svMoi.maSV);
        for(int i = 0; i < *soLuong; i++) {
            if(strcmp(ds[i].maSV, svMoi.maSV) == 0) {
                maTrung = 1;
                printf("MaSV bi trung, nhap lai!\n");
                break;
            }
        }
    } while(maTrung);

    printf("Ho ten: ");
    getchar();
    fgets(svMoi.hoTen, sizeof(svMoi.hoTen), stdin);
    svMoi.hoTen[strcspn(svMoi.hoTen, "\n")] = '\0';

    printf("Nganh hoc: ");
    fgets(svMoi.nganhHoc, sizeof(svMoi.nganhHoc), stdin);
    svMoi.nganhHoc[strcspn(svMoi.nganhHoc, "\n")] = '\0';

    printf("Nam nhap hoc: ");
    scanf("%d", &svMoi.namNhapHoc);

    ds[*soLuong] = svMoi;
    (*soLuong)++;
}

void sapXepTheoNam(struct SinhVien ds[], int soLuong) {
    for(int i = 0; i < soLuong-1; i++) {
        for(int j = i+1; j < soLuong; j++) {
            if(ds[i].namNhapHoc > ds[j].namNhapHoc) {
                struct SinhVien tam = ds[i];
                ds[i] = ds[j];
                ds[j] = tam;
            }
        }
    }
    printf("Danh sach da sap xep tang dan theo nam nhap hoc.\n");
}

int main() {
    system("cls");
    struct SinhVien ds[100];
    int soLuong = 0, luaChon;

    do {
        printf("\n===== MENU =====\n");
        printf("1. Nhap thong tin sinh vien\n");
        printf("2. Hien thi tat ca sinh vien\n");
        printf("3. Tim sinh vien theo nganh\n");
        printf("4. Them mot sinh vien moi\n");
        printf("5. Sap xep sinh vien theo nam nhap hoc\n");
        printf("0. Thoat\n");
        printf("Lua chon: ");
        scanf("%d", &luaChon);

        switch(luaChon) {
            case 1: nhapSinhVien(ds, &soLuong); break;
            case 2: hienThiSinhVien(ds, soLuong); break;
            case 3: timTheoNganh(ds, soLuong); break;
            case 4: themSinhVien(ds, &soLuong); break;
            case 5: sapXepTheoNam(ds, soLuong); hienThiSinhVien(ds, soLuong); break;
            case 0: printf("Dang thoat chuong trinh...\n"); break;
            default: printf("Lua chon khong hop le!\n");
        }
    } while(luaChon != 0);

    return 0;
}
