from rest_framework import serializers

from .models import (
    Agent,
    Conversation,
    KnowledgeBase,
    KnowledgeDocument,
    MCPServer,
    MCPTool,
    Message,
    Pipeline,
    PipelineStep,
    Project,
    Task,
)


class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ["id", "name", "description", "created_at"]
        read_only_fields = ["id", "created_at"]


class KnowledgeBaseSerializer(serializers.ModelSerializer):
    project = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = KnowledgeBase
        fields = ["id", "project", "name", "description", "created_at"]
        read_only_fields = ["id", "project", "created_at"]


class KnowledgeDocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeDocument
        fields = [
            "id",
            "knowledge_base",
            "file",
            "mime_type",
            "status",
            "meta",
            "created_at",
        ]
        read_only_fields = ["id", "mime_type", "status", "meta", "created_at"]


class AgentSerializer(serializers.ModelSerializer):
    resolved_model_name = serializers.CharField(read_only=True)
    warning = serializers.CharField(read_only=True, allow_null=True)

    class Meta:
        model = Agent
        fields = [
            "id",
            "project",
            "name",
            "system_prompt",
            "model_name",
            "resolved_model_name",
            "warning",
            "temperature",
            "max_tokens",
            "is_primary",
            "is_active",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "resolved_model_name", "created_at", "updated_at"]


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "sender", "content", "meta", "created_at"]
        read_only_fields = ["id", "sender", "meta", "created_at"]


class ConversationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = [
            "id",
            "project",
            "agent",
            "title",
            "is_archived",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class ConversationDetailSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = [
            "id",
            "project",
            "agent",
            "title",
            "is_archived",
            "created_at",
            "updated_at",
            "messages",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "messages"]


class MCPServerSerializer(serializers.ModelSerializer):
    class Meta:
        model = MCPServer
        fields = [
            "id",
            "name",
            "description",
            "command",
            "command_args",
            "transport",
            "is_active",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class MCPToolSerializer(serializers.ModelSerializer):
    server = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = MCPTool
        fields = [
            "id",
            "server",
            "name",
            "description",
            "input_schema",
            "output_schema",
            "is_active",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "server", "created_at", "updated_at"]


class PipelineStepSerializer(serializers.ModelSerializer):
    class Meta:
        model = PipelineStep
        fields = [
            "id",
            "order",
            "type",
            "agent",
            "tool",
            "config",
        ]
        read_only_fields = ["id"]


class PipelineSerializer(serializers.ModelSerializer):
    steps = PipelineStepSerializer(many=True, read_only=True)

    class Meta:
        model = Pipeline
        fields = [
            "id",
            "project",
            "owner_agent",
            "name",
            "description",
            "is_active",
            "created_at",
            "updated_at",
            "steps",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "steps"]


class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Task
        fields = [
            "id",
            "pipeline",
            "created_by_agent",
            "input_payload",
            "result_payload",
            "status",
            "current_step_index",
            "error_message",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "pipeline",
            "created_by_agent",
            "result_payload",
            "status",
            "current_step_index",
            "error_message",
            "created_at",
            "updated_at",
        ]
