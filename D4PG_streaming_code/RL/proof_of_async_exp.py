import asyncio

class AsyncIterable:
    def __init__(self, limit):
        self.limit = limit

    async def __aiter__(self):
        for i in range(self.limit):
            yield i
            await asyncio.sleep(0.1)  # 模拟异步操作

class AsyncIteratorWrapper(AsyncIterable):
    def __init__(self, limit):
        super().__init__(limit)

    async def __aiter__(self):
        async for item in super().__aiter__():
            yield item

async def test_anext():
    async_iterable_wrapper = AsyncIteratorWrapper(AsyncIterable(5))
    try:
        while True:
            item = await async_iterable_wrapper.__anext__()
            print(item)
    except StopIteration:
        pass

# 运行测试函数
asyncio.run(test_anext())
