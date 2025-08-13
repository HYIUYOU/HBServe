"""
RPC客户端，用于CPU端调度器与GPU实例通信
"""
import grpc
import asyncio
import logging
from typing import Dict, Any, Optional, List

# 定义RPC消息结构（简化版）
class RpcMessage:
    """RPC消息基类"""
    def __init__(self, message_type: str, data: Dict[str, Any]):
        self.message_type = message_type
        self.data = data
        
    def to_json(self) -> Dict[str, Any]:
        return {
            "message_type": self.message_type,
            "data": self.data
        }
        
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'RpcMessage':
        return cls(
            message_type=json_data["message_type"],
            data=json_data["data"]
        )


class RpcClient:
    """RPC客户端"""
    
    def __init__(self, instance_id: str, host: str, port: int, timeout: float = 30.0):
        self.instance_id = instance_id
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        # 连接状态
        self.connected = False
        self.channel = None
        self.reader = None
        self.writer = None
        
    async def connect(self):
        """连接到GPU实例"""
        try:
            # 使用TCP连接（简化版，实际可用gRPC）
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port
            )
            self.connected = True
            self.logger.info(f"成功连接到实例 {self.instance_id} ({self.host}:{self.port})")
            
        except Exception as e:
            self.logger.error(f"连接实例 {self.instance_id} 失败: {e}")
            raise
            
    async def close(self):
        """关闭连接"""
        try:
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()
            self.connected = False
            self.logger.info(f"已关闭与实例 {self.instance_id} 的连接")
            
        except Exception as e:
            self.logger.error(f"关闭连接失败: {e}")
            
    async def _send_message(self, message: RpcMessage) -> Optional[Dict[str, Any]]:
        """发送RPC消息"""
        if not self.connected:
            raise RuntimeError(f"未连接到实例 {self.instance_id}")
            
        try:
            # 序列化消息
            import json
            message_json = json.dumps(message.to_json()).encode('utf-8')
            message_length = len(message_json)
            
            # 发送消息长度（4字节）+ 消息内容
            self.writer.write(message_length.to_bytes(4, byteorder='big'))
            self.writer.write(message_json)
            await self.writer.drain()
            
            # 接收响应
            response_length_bytes = await self.reader.readexactly(4)
            response_length = int.from_bytes(response_length_bytes, byteorder='big')
            
            response_data = await self.reader.readexactly(response_length)
            response_json = json.loads(response_data.decode('utf-8'))
            
            return response_json
            
        except Exception as e:
            self.logger.error(f"发送消息失败: {e}")
            return None
            
    async def get_status(self) -> Dict[str, Any]:
        """获取实例状态"""
        message = RpcMessage("get_status", {})
        response = await self._send_message(message)
        
        if response:
            return response.get("data", {})
        else:
            return {"available": False, "error": "通信失败"}
            
    async def prefill(self, 
                     request_id: str,
                     token_ids: List[int],
                     sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """发送prefill请求"""
        message = RpcMessage("prefill", {
            "request_id": request_id,
            "token_ids": token_ids,
            "sampling_params": sampling_params
        })
        
        response = await self._send_message(message)
        
        if response:
            return response.get("data", {})
        else:
            return {"success": False, "error": "通信失败"}
            
    async def transfer_kv_cache(self, 
                               request_id: str,
                               target_instance: str) -> Dict[str, Any]:
        """请求传输KV缓存"""
        message = RpcMessage("transfer_kv_cache", {
            "request_id": request_id,
            "target_instance": target_instance
        })
        
        response = await self._send_message(message)
        
        if response:
            return response.get("data", {})
        else:
            return {"success": False, "error": "通信失败"}
            
    async def get_transfer_status(self, request_id: str) -> Dict[str, Any]:
        """获取KV传输状态"""
        message = RpcMessage("get_transfer_status", {
            "request_id": request_id
        })
        
        response = await self._send_message(message)
        
        if response:
            return response.get("data", {})
        else:
            return {"status": "unknown", "error": "通信失败"}
            
    async def start_decode(self, 
                          request_id: str,
                          sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """开始decode"""
        message = RpcMessage("start_decode", {
            "request_id": request_id,
            "sampling_params": sampling_params
        })
        
        response = await self._send_message(message)
        
        if response:
            return response.get("data", {})
        else:
            return {"success": False, "error": "通信失败"}
            
    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """获取请求状态"""
        message = RpcMessage("get_request_status", {
            "request_id": request_id
        })
        
        response = await self._send_message(message)
        
        if response:
            return response.get("data", {})
        else:
            return {"status": "unknown", "error": "通信失败"}
            
    async def chunked_prefill(self, 
                             request_id: str,
                             token_ids: List[int],
                             chunk_size: int,
                             sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        """分块prefill请求"""
        message = RpcMessage("chunked_prefill", {
            "request_id": request_id,
            "token_ids": token_ids,
            "chunk_size": chunk_size,
            "sampling_params": sampling_params
        })
        
        response = await self._send_message(message)
        
        if response:
            return response.get("data", {})
        else:
            return {"success": False, "error": "通信失败"}
            
    async def get_stats(self) -> Dict[str, Any]:
        """获取实例统计信息"""
        message = RpcMessage("get_stats", {})
        response = await self._send_message(message)
        
        if response:
            return response.get("data", {})
        else:
            return {"error": "通信失败"}


class RpcServer:
    """RPC服务器，运行在GPU实例上"""
    
    def __init__(self, host: str, port: int, message_handler):
        self.host = host
        self.port = port
        self.message_handler = message_handler
        self.logger = logging.getLogger(__name__)
        self.server = None
        self.running = False
        
    async def start(self):
        """启动RPC服务器"""
        try:
            self.server = await asyncio.start_server(
                self._handle_client,
                self.host,
                self.port
            )
            self.running = True
            
            self.logger.info(f"RPC服务器启动在 {self.host}:{self.port}")
            
            async with self.server:
                await self.server.serve_forever()
                
        except Exception as e:
            self.logger.error(f"RPC服务器启动失败: {e}")
            raise
            
    async def stop(self):
        """停止RPC服务器"""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            self.running = False
            self.logger.info("RPC服务器已停止")
            
        except Exception as e:
            self.logger.error(f"停止RPC服务器失败: {e}")
            
    async def _handle_client(self, reader, writer):
        """处理客户端连接"""
        client_addr = writer.get_extra_info('peername')
        self.logger.debug(f"新客户端连接: {client_addr}")
        
        try:
            while True:
                # 读取消息长度
                length_data = await reader.readexactly(4)
                message_length = int.from_bytes(length_data, byteorder='big')
                
                # 读取消息内容
                message_data = await reader.readexactly(message_length)
                
                import json
                message_json = json.loads(message_data.decode('utf-8'))
                message = RpcMessage.from_json(message_json)
                
                # 处理消息
                response = await self.message_handler.handle_message(message)
                
                # 发送响应
                response_json = json.dumps(response).encode('utf-8')
                response_length = len(response_json)
                
                writer.write(response_length.to_bytes(4, byteorder='big'))
                writer.write(response_json)
                await writer.drain()
                
        except asyncio.IncompleteReadError:
            # 客户端断开连接
            self.logger.debug(f"客户端断开连接: {client_addr}")
        except Exception as e:
            self.logger.error(f"处理客户端消息失败: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except:
                pass
