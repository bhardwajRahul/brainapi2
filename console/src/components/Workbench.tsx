import type { ReactNode } from "react";
import {
  Button,
  CodeBlock,
  PageHeader,
  PageHeaderActions,
  PageHeaderContent,
  PageHeaderDescription,
  PageHeaderTitle,
} from "lumen-ui-kit";
import { CloseIcon, Icon } from "lumen-ui-kit/icons";

export function InlineField({
  id,
  label,
  children,
  className = "",
}: {
  id: string;
  label: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`flex min-w-0 flex-row flex-wrap items-center gap-2 ${className}`.trim()}
    >
      <label
        htmlFor={id}
        className="mb-0 shrink-0 whitespace-nowrap text-sm font-medium text-lumen-muted-foreground"
      >
        {label}
      </label>
      <div className="min-w-0 [&_[data-slot=input]]:h-9 [&_[data-slot=select]]:h-9 [&_input]:h-9 [&_select]:h-9">
        {children}
      </div>
    </div>
  );
}

export function HeaderActions({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <PageHeaderActions
      className={`flex flex-wrap items-center gap-2 ${className}`.trim()}
    >
      {children}
    </PageHeaderActions>
  );
}

export function FilterBar({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`flex flex-wrap items-center gap-2 ${className}`.trim()}
      role="group"
    >
      {children}
    </div>
  );
}

export function Workbench({
  title,
  description,
  actions,
  toolbar,
  children,
  inspector,
  flush = false,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  toolbar?: ReactNode;
  children: ReactNode;
  inspector?: ReactNode;
  flush?: boolean;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        className={
          flush
            ? "border-b border-lumen-border px-4 py-3 sm:px-6"
            : "px-4 pt-6 sm:px-6"
        }
      >
        <PageHeader className="sm:items-center">
          <PageHeaderContent>
            <PageHeaderTitle>{title}</PageHeaderTitle>
            {description ? (
              <PageHeaderDescription>{description}</PageHeaderDescription>
            ) : null}
          </PageHeaderContent>
          {actions ? <HeaderActions>{actions}</HeaderActions> : null}
        </PageHeader>
        {toolbar ? <div className="mt-4">{toolbar}</div> : null}
      </div>

      <div
        className={
          flush
            ? "flex min-h-0 flex-1"
            : "mt-4 flex min-h-0 flex-1 gap-0 px-4 pb-6 sm:px-6"
        }
      >
        <div
          className={
            flush
              ? "min-h-0 min-w-0 flex-1 overflow-auto"
              : "min-h-0 min-w-0 flex-1 overflow-auto border border-lumen-border bg-lumen-surface"
          }
        >
          {children}
        </div>
        {inspector}
      </div>
    </div>
  );
}

export function DetailInspector({
  title,
  subtitle,
  onClose,
  children,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <aside
      className="flex w-full shrink-0 flex-col border-t border-lumen-border bg-lumen-surface xl:w-[var(--lumen-layout-rail-width)] xl:border-l xl:border-t-0"
      aria-label="Record detail"
    >
      <div className="flex items-center justify-between gap-3 border-b border-lumen-border px-4 py-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-lumen-foreground">
            {title}
          </h2>
          {subtitle ? (
            <p className="mt-0.5 truncate font-mono text-[11px] text-lumen-muted-foreground">
              {subtitle}
            </p>
          ) : null}
        </div>
        <Button
          type="button"
          size="icon"
          variant="ghost"
          aria-label="Close detail panel"
          onClick={onClose}
        >
          <Icon source={CloseIcon} />
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-4">{children}</div>
    </aside>
  );
}

export function JsonInspector({
  title,
  subtitle,
  value,
  onClose,
}: {
  title: ReactNode;
  subtitle?: ReactNode;
  value: unknown;
  onClose: () => void;
}) {
  return (
    <DetailInspector title={title} subtitle={subtitle} onClose={onClose}>
      <CodeBlock className="text-[11px] leading-relaxed">
        {JSON.stringify(value, null, 2)}
      </CodeBlock>
    </DetailInspector>
  );
}

export function PageFrame({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`px-4 py-6 sm:px-6 ${className}`.trim()}>{children}</div>
  );
}
