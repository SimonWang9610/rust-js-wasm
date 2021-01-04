- IndexedDB：`/node/www/tcp/js/files.js`
  - `open` returns an `IDBOpenDBRequest` object with a result(success) or error value
  - The result of `IDBRequest` is an instance of an `IDBDatabase`

        var request = window.indexedDb.open("TestDatabase", 1);


- 整体设计思路：
  - 页面载入的时候检查`indexedDB`是否存储的有相应的模型参数文件
    - 有。提示模型准备完毕。
    - 没有
      - `fetch`相应的模型文件
      - 确认`indexedDB`版本号，对版本号进行`upgrade`
      - 建立新的`Store`,将模型参数存入database

​		



