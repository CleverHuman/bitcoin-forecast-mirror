import { AppCardItem } from "./custom/app-card-item";

import { Skeleton } from "@/registry/default/ui/skeleton";

export default function GridListSkeleton() {
    return Array(6).fill(0).map((_, index) => (
        <AppCardItem key={`shimmer-${index}`} className="p-0 overflow-hidden bg-inherit">
          <div className="flex flex-col h-full">
            <div className="px-4 py-2 border-t">
              <div className="flex items-center justify-between">
                <div className="flex flex-col min-w-0 mr-2 space-y-2">
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-3 w-24" />
                </div>
                <Skeleton className="h-8 w-8" />
              </div>
            </div>
            <div className="flex-1 p-4">
              <Skeleton className="h-16" />
            </div>
          </div>
        </AppCardItem>
      ))
}
