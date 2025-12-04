from rest_framework import serializers

from core.models import KnowledgeChangeLog, KnowledgeSource


class RAGSourceSerializer(serializers.ModelSerializer):
    relative_path = serializers.CharField(source="path", read_only=True)
    hash = serializers.CharField(source="content_hash", read_only=True)
    error_message = serializers.CharField(source="last_error", read_only=True)

    class Meta:
        model = KnowledgeSource
        fields = (
            "id",
            "relative_path",
            "status",
            "mime_type",
            "hash",
            "updated_at",
            "error_message",
        )
        read_only_fields = fields


class RAGChangeSerializer(serializers.ModelSerializer):
    source_path = serializers.CharField(source="source.path", read_only=True)

    class Meta:
        model = KnowledgeChangeLog
        fields = (
            "id",
            "source_path",
            "change_type",
            "semantic_summary",
            "diff_text",
            "new_facts",
            "removed_facts",
            "created_at",
        )
        read_only_fields = fields
