struct dataElem {
  int prop1;
  int prop2;
  char *name;
}

void launch(dataElem *elem)
{
  dataElem *d_elem;
  char *d_name;

  int namelen = strlen(elem->name) + 1;

  // Выделяем память под структуру и под поле name
  cudaMalloc(&d_elem, sizeof(dataElem));
  cudaMalloc(&d_name, namelen);

  // Отдельно копируем структуру, значение поля name и значение указателя на поле name 
  cudaMemcpy(d_elem, elem, sizeof(dataElem), cudaMemcpyHostToDevice);
  cudaMemcpy(d_name, elem->name, namelen, cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_elem->name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

  // Уффф
  Kernel<<< ... >>>(d_elem);
}

//Вместо 1000 слов
void launch_UV(dataElem *elem)
{
  kernel<<< ... >>>(elem);
}

/*-----------------------*/
class Managed 
{
public:
  void *operator new(size_t len)
  {
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
  }

  void operator delete(void *ptr) 
  {
    cudaDeviceSynchronize();
    cudaFree(ptr);
  }
};

// Наследование от Managed позволяет ссылку по указателю
class String : public Managed 
{
  int length;
  char *data;

public:
  // Unified memory позволяет конструктор копирования
  String (const String &s) 
  {
    length = s.length;
    cudaMallocManaged(&data, length);
    memcpy(data, s.data, length);
  }

  // ...
};